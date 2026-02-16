use nalgebra::{self as na, SMatrix, SVector};

/// This Trait is used to abstract over `f32` and `f64` in the codebase.
/// You can use this in trait bounds to write generic code.
pub fn symmetric_eigen<const N: usize>(
    matrix: na::SMatrix<f64, N, N>,
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
