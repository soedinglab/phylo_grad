use nalgebra::{self as na, SMatrix, SVector};

/// Eigendecomposition of a symmetric matrix, only the lower triangle is read.
/// Returns None if the QR iteration did not converge.
///
/// nalgebra before 0.34.2 returns wrong eigenvectors for many matrices
/// (https://github.com/dimforge/nalgebra/issues/1109), so Cargo.toml requires at least 0.34.2.
pub fn symmetric_eigen<const N: usize>(
    matrix: na::SMatrix<f64, N, N>,
) -> Option<(SVector<f64, N>, SMatrix<f64, N, N>)> {
    // nalgebra's symmetric_eigen is not available for generic SMatrix<_, N, N>, so we go through DMatrix
    let dynamic_matrix = na::DMatrix::from_column_slice(N, N, matrix.as_slice());

    // The iteration limit only guards against non termination (https://github.com/dimforge/nalgebra/issues/1025),
    // a converging decomposition needs a few iterations per eigenvalue
    let eigen = dynamic_matrix.try_symmetric_eigen(f64::EPSILON, 100 * N + 100)?;

    if eigen.eigenvalues.iter().any(|x| !x.is_finite()) {
        return None;
    }

    Some((
        SVector::<f64, N>::from_column_slice(eigen.eigenvalues.as_slice()),
        SMatrix::<f64, N, N>::from_column_slice(eigen.eigenvectors.as_slice()),
    ))
}

/// exp(x) with at most 1 ulp difference to the libm exp, written without branches so it vectorizes.
/// Results below exp(-708) (~1e-308) are flushed to zero instead of being subnormal.
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    let xc = x.clamp(-708.0, 709.0);
    let k = (xc * LOG2E).round();
    let r = (xc - k * LN2_HI) - k * LN2_LO;
    // Taylor series up to r^13, the error for |r| <= ln2 / 2 is below 1e-17
    let mut p = 1.0 / 6227020800.0;
    for c in [
        1.0 / 479001600.0,
        1.0 / 39916800.0,
        1.0 / 3628800.0,
        1.0 / 362880.0,
        1.0 / 40320.0,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = p * r + c;
    }
    // k + 1023 is in the low mantissa bits of k + 1023 + 2^52, shifting them into the exponent gives 2^k
    // without a float to int conversion, which does not vectorize on AVX2
    let scale = f64::from_bits((k + (1023.0 + 4503599627370496.0)).to_bits() << 52);
    if x < -708.0 { 0.0 } else { p * scale }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// max of the relative reconstruction error |M V - V diag(lambda)| / |M| and the orthogonality error |V^T V - I|
    fn decomposition_error<const N: usize>(matrix: &SMatrix<f64, N, N>) -> f64 {
        let (eigenvalues, eigenvectors) = symmetric_eigen(*matrix).expect("decomposition failed");
        let reconstruction = (matrix * eigenvectors
            - eigenvectors * SMatrix::<f64, N, N>::from_diagonal(&eigenvalues))
        .norm()
            / matrix.norm().max(f64::MIN_POSITIVE);
        let orthogonality =
            (eigenvectors.transpose() * eigenvectors - SMatrix::<f64, N, N>::identity()).norm();
        let error = reconstruction.max(orthogonality);
        if error.is_finite() {
            error
        } else {
            f64::INFINITY
        }
    }

    /// Symmetrized reversible rate matrix like in compute_param_data.
    /// Rates and pi are log uniform over e^(+-spread/2), with tiny_pi a fraction of the states gets pi = 1e-12
    fn reversible<const N: usize>(
        rng: &mut impl rand::Rng,
        spread: f64,
        tiny_pi: bool,
    ) -> SMatrix<f64, N, N> {
        let mut pi =
            SVector::<f64, N>::from_fn(|_, _| (spread * (rng.random::<f64>() - 0.5)).exp());
        if tiny_pi {
            pi = pi.map(|p| if rng.random::<f64>() < 0.3 { 1e-12 } else { p });
        }
        pi /= pi.sum();
        let sqrt_pi = pi.map(f64::sqrt);

        let mut matrix = SMatrix::<f64, N, N>::zeros();
        for i in 0..N {
            for j in 0..i {
                let rate = (spread * (rng.random::<f64>() - 0.5)).exp();
                matrix[(i, j)] = rate;
                matrix[(j, i)] = rate;
            }
        }
        for i in 0..N {
            matrix[(i, i)] = -(0..N)
                .filter(|&j| j != i)
                .map(|j| matrix[(i, j)] * sqrt_pi[j] / sqrt_pi[i])
                .sum::<f64>();
        }
        matrix
    }

    /// Jukes-Cantor with a symmetric perturbation of random size between 1 and 1e-16, so the eigenvalues are (nearly) degenerate
    fn jukes_cantor_perturbed<const N: usize>(rng: &mut impl rand::Rng) -> SMatrix<f64, N, N> {
        let scale = 10f64.powf(-16.0 * rng.random::<f64>());
        let perturbation =
            SMatrix::<f64, N, N>::from_fn(|_, _| scale * (rng.random::<f64>() - 0.5));
        SMatrix::<f64, N, N>::from_fn(|i, j| if i == j { 1.0 - N as f64 } else { 1.0 })
            + perturbation
            + perturbation.transpose()
    }

    /// Entries spanning 16 orders of magnitude
    fn graded<const N: usize>(rng: &mut impl rand::Rng) -> SMatrix<f64, N, N> {
        let grades =
            SVector::<f64, N>::from_fn(|_, _| 10f64.powf(16.0 * rng.random::<f64>() - 8.0));
        let matrix = SMatrix::<f64, N, N>::from_fn(|i, j| {
            (rng.random::<f64>() - 0.5) * (grades[i] * grades[j]).sqrt()
        });
        matrix + matrix.transpose()
    }

    /// Tridiagonal with integer diagonal and off diagonals down to 1e-20, decouples into many small blocks
    fn tridiagonal_tiny_off_diagonal<const N: usize>(
        rng: &mut impl rand::Rng,
    ) -> SMatrix<f64, N, N> {
        let mut matrix = SMatrix::<f64, N, N>::zeros();
        for i in 0..N {
            matrix[(i, i)] = (3.0 * rng.random::<f64>()).floor() + 1e-10 * rng.random::<f64>();
            if i > 0 {
                let off_diagonal = 10f64.powf(-20.0 * rng.random::<f64>());
                matrix[(i, i - 1)] = off_diagonal;
                matrix[(i - 1, i)] = off_diagonal;
            }
        }
        matrix
    }

    /// Random rotation of a spectrum with clusters of eigenvalues 1e-13 apart
    fn clustered<const N: usize>(rng: &mut impl rand::Rng) -> SMatrix<f64, N, N> {
        let random = na::DMatrix::<f64>::from_fn(N, N, |_, _| rng.random::<f64>() - 0.5);
        let rotation = SMatrix::<f64, N, N>::from_column_slice(random.qr().q().as_slice());
        let spectrum = SVector::<f64, N>::from_fn(|_, _| {
            (4.0 * rng.random::<f64>()).floor() + 1e-13 * (rng.random::<f64>() - 0.5)
        });
        rotation * SMatrix::<f64, N, N>::from_diagonal(&spectrum) * rotation.transpose()
    }

    fn check_all_families<const N: usize>(samples: usize) {
        type Generator<const N: usize> = fn(&mut rand::rngs::StdRng) -> SMatrix<f64, N, N>;
        let families: [(&str, Generator<N>); 7] = [
            ("reversible mild", |rng| reversible(rng, 4.0, false)),
            ("reversible wide", |rng| reversible(rng, 16.0, false)),
            ("reversible tiny pi", |rng| reversible(rng, 20.0, true)),
            ("Jukes-Cantor perturbed", jukes_cantor_perturbed),
            ("graded", graded),
            (
                "tridiagonal tiny off diagonal",
                tridiagonal_tiny_off_diagonal,
            ),
            ("clustered", clustered),
        ];
        for (seed, (name, generate)) in families.iter().enumerate() {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
            for sample in 0..samples {
                let matrix = generate(&mut rng);
                let error = decomposition_error(&matrix);
                assert!(
                    error < 1e-12,
                    "{name}, N = {N}, sample {sample}: decomposition error {error:e}\n{matrix}"
                );
            }
        }
    }

    #[test]
    fn symmetric_eigen_4_states() {
        check_all_families::<4>(2000);
    }

    #[test]
    fn symmetric_eigen_20_states() {
        check_all_families::<20>(500);
    }

    #[test]
    fn symmetric_eigen_64_states() {
        check_all_families::<64>(30);
    }

    /// Matrix from https://github.com/dimforge/nalgebra/issues/1109
    #[test]
    fn symmetric_eigen_nalgebra_issue_1109() {
        let matrix = na::Matrix4::new(
            -19884.07, -10.07188, 11.277279, -188560.63, //
            -10.07188, 12.518197, 1.3770627, -102.97504, //
            11.277279, 1.3770627, 14.587362, 113.26099, //
            -188560.63, -102.97504, 113.26099, -1788112.3,
        );
        assert!(decomposition_error(&matrix) < 1e-12);
    }

    #[test]
    fn symmetric_eigen_diagonal_and_zero() {
        let diagonal =
            SMatrix::<f64, 20, 20>::from_diagonal(&SVector::from_fn(|i, _| i as f64 - 10.0));
        assert!(decomposition_error(&diagonal) < 1e-14);
        let (eigenvalues, _) = symmetric_eigen(SMatrix::<f64, 20, 20>::zeros()).unwrap();
        assert!(eigenvalues.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn symmetric_eigen_non_finite_input() {
        let mut matrix = SMatrix::<f64, 4, 4>::identity();
        matrix[(1, 0)] = f64::NAN;
        matrix[(0, 1)] = f64::NAN;
        assert!(symmetric_eigen(matrix).is_none());
    }

    #[test]
    fn exp_matches_libm() {
        let mut state = 1u64;
        for _ in 0..1_000_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = -700.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) + 5.0;
            let (fast, libm) = (super::exp(x), x.exp());
            let ulps = (fast.to_bits() as i64 - libm.to_bits() as i64).unsigned_abs();
            assert!(ulps <= 1, "exp({x}) = {fast} vs {libm}");
        }
        assert_eq!(super::exp(0.0), 1.0);
        assert_eq!(super::exp(-1000.0), 0.0);
        assert_eq!(super::exp(f64::NEG_INFINITY), 0.0);
        assert!(super::exp(f64::NAN).is_nan());
    }
}
