use std::iter::Sum;
use logsumexp::LogSumExp;
use nalgebra::{self as na, SMatrix, SVector};

/// This Trait is used to abstract over `f32` and `f64` in the codebase.
/// You can use this in trait bounds to write generic code.
pub trait FloatTrait
where
    Self: num_traits::Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + na::Scalar
        + std::marker::Sync
        + Into<f64>
        + Sum
        + na::RealField,
{
    const EPS_LOG: Self;
    const MIN_SQRT_PI: Self;
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self;
    fn from_f64(f: f64) -> Self;
    fn symmetric_eigen<const N: usize>(
        matrix: na::SMatrix<Self, N, N>,
    ) -> Option<(SVector<Self, N>, SMatrix<Self, N, N>)>;
}

impl FloatTrait for f32 {
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f as f32
    }
    const EPS_LOG: Self = 1e-20;
    const MIN_SQRT_PI: Self = 1e-10;
    fn symmetric_eigen<const N: usize>(
        matrix: na::SMatrix<Self, N, N>,
    ) -> Option<(SVector<f32, N>, SMatrix<f32, N, N>)> {
        let faer_matirx = faer::mat::MatRef::from_column_major_array(&matrix.data.0);

        let eigen_decomposition = faer_matirx.self_adjoint_eigen(faer::Side::Upper);

        if let Ok(result) = eigen_decomposition {
            let eigenvalues = result.S();
            let eigenvectors = result.U();

            let mut eigenvalues_na = SVector::<f32, N>::zeros();
            for i in 0..N {
                eigenvalues_na[i] = eigenvalues[i];
            }

            let mut eigenvectors_na = SMatrix::<f32, N, N>::zeros();
            for i in 0..N {
                for j in 0..N {
                    eigenvectors_na[(i, j)] = eigenvectors[(i, j)];
                }
            }

            Some((eigenvalues_na, eigenvectors_na))
        } else {
            None
        }

    }
}
impl FloatTrait for f64 {
    fn logsumexp<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        LogSumExp::ln_sum_exp(iter)
    }
    fn from_f64(f: f64) -> Self {
        f
    }
    const EPS_LOG: Self = 1e-100;
    const MIN_SQRT_PI: Self = 1e-10;
    fn symmetric_eigen<const N: usize>(
        matrix: na::SMatrix<Self, N, N>,
    ) -> Option<(SVector<f64, N>, SMatrix<f64, N, N>)> {
        let faer_matirx = faer::mat::MatRef::from_column_major_array(&matrix.data.0);

        let eigen_decomposition = faer_matirx.self_adjoint_eigen(faer::Side::Upper);

        if let Ok(result) = eigen_decomposition {
            let eigenvalues = result.S();
            let eigenvectors = result.U();

            let mut eigenvalues_na = SVector::<f64, N>::zeros();
            for i in 0..N {
                eigenvalues_na[i] = eigenvalues[i];
            }

            let mut eigenvectors_na = SMatrix::<f64, N, N>::zeros();
            for i in 0..N {
                for j in 0..N {
                    eigenvectors_na[(i, j)] = eigenvectors[(i, j)];
                }
            }

            Some((eigenvalues_na, eigenvectors_na))
        } else {
            None
        }
    }
}
