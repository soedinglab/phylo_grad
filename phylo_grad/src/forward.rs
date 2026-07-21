use nalgebra as na;

/// Forward data precomputed before the forward pass
pub struct ForwardData<const DIM: usize> {
    pub model_edge_data: Vec<ModelEdgeData<DIM>>,
}

impl<const DIM: usize> ForwardData<DIM> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            model_edge_data: Vec::with_capacity(capacity),
        }
    }
}

/// Data precomputed for each edge. Depends only on the Q matrix and the edge length
#[derive(Debug)]
pub struct ModelEdgeData<const DIM: usize> {
    /// exp(t * lambda_i) for the DIM many eigenvalues of Q
    pub exp_t_lambda: na::SVector<f64, DIM>,
}

/// Precomputed values from the model (S and sqrt_pi)
#[derive(Debug)]
pub struct ParamPrecomp<const DIM: usize> {
    /// S
    pub symmetric_matrix: na::SMatrix<f64, DIM, DIM>,
    /// sqrt_pi
    pub sqrt_pi: na::SVector<f64, DIM>,
    /// 1/sqrt_pi
    pub sqrt_pi_recip: na::SVector<f64, DIM>,
    /// Eigenvalues of S
    pub eigenvalues: na::SVector<f64, DIM>,
    /// A in the paper
    pub V_pi: na::SMatrix<f64, DIM, DIM>,
    /// A^-1 in the paper
    pub V_pi_inv: na::SMatrix<f64, DIM, DIM>,
    /// Q
    pub Q: na::SMatrix<f64, DIM, DIM>,
}

/// In-place multiplication by a diagonal matrix on the left
pub fn diag_times_assign<I, const N: usize>(
    mut matrix: na::SMatrixViewMut<f64, N, N>,
    diagonal_entries: I,
) where
    I: Iterator<Item = f64>,
{
    for (mut row, scale) in std::iter::zip(matrix.row_iter_mut(), diagonal_entries) {
        row *= scale;
    }
}

/// In-place multiplication by a diagonal matrix on the right
pub fn times_diag_assign<I, const N: usize>(
    mut matrix: na::SMatrixViewMut<f64, N, N>,
    diagonal_entries: I,
) where
    I: Iterator<Item = f64>,
{
    for (mut col, scale) in std::iter::zip(matrix.column_iter_mut(), diagonal_entries) {
        col *= scale;
    }
}

/// Precomputes things out of S and sqrt_pi
/// Returns None if the eigenvalues are too large or the diagonalization failed, this can happen with extreme values
pub fn compute_param_data<const DIM: usize>(
    S: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
) -> Option<ParamPrecomp<DIM>> {
    let sqrt_pi_recip = sqrt_pi.map(|x| f64::recip(f64::max(x, f64::MIN_POSITIVE)));

    // Read only the upper triangle of S and make it symmetric
    let mut S_symmetric = S.clone_owned();
    for i in 0..DIM {
        for j in 0..i {
            S_symmetric[(i, j)] = S_symmetric[(j, i)];
        }
    }

    /* rate_matrix = diag(sqrt_pi_recip) * S_output * diag(sqrt_pi) */
    let mut rate_matrix = S_symmetric.clone_owned();
    diag_times_assign(rate_matrix.as_view_mut(), sqrt_pi_recip.iter().copied());
    times_diag_assign(rate_matrix.as_view_mut(), sqrt_pi.iter().copied());

    for i in 0..DIM {
        rate_matrix[(i, i)] -= rate_matrix.row(i).sum();
    }

    // S_sym has unspecified diagonal elements, so we put the correct ones from the rate matrix
    for i in 0..DIM {
        S_symmetric[(i, i)] = rate_matrix[(i, i)];
    }

    let (eigenvalues, eigenvectors) = crate::numerics::symmetric_eigen(S_symmetric)?;

    let mut V_pi = eigenvectors;
    diag_times_assign(V_pi.as_view_mut(), sqrt_pi_recip.iter().copied());

    let mut V_pi_inv = eigenvectors.transpose();
    times_diag_assign(V_pi_inv.as_view_mut(), sqrt_pi.iter().copied());

    Some(ParamPrecomp {
        symmetric_matrix: S_symmetric,
        sqrt_pi: sqrt_pi.clone_owned(),
        sqrt_pi_recip,
        eigenvalues,
        V_pi,
        V_pi_inv,
        Q: rate_matrix,
    })
}

fn precompute_model_edge_data<const DIM: usize>(
    param: &ParamPrecomp<DIM>,
    distance: f64,
) -> ModelEdgeData<DIM> {
    let exp_t_lambda = param.eigenvalues.map(|lam| f64::exp(lam * distance));

    ModelEdgeData { exp_t_lambda }
}

pub fn forward_data_precompute_param<const DIM: usize>(
    param: &ParamPrecomp<DIM>,
    distances: &[f64],
) -> Vec<ModelEdgeData<DIM>> {
    distances
        .iter()
        .map(|dist| precompute_model_edge_data(param, *dist))
        .collect()
}

/// adds the log_p of the children to the log_p of the parent
/// Main part of the Felsenstein in Forward
/// log_p are the partial log likelihoods, they start with the leave nodes initialized. This function takes 2 computed log_p vectors
/// and writes the combined result in the parent log_p vector
/// Offsets are scaling factors to prevent underflow. offsets[i] = 10 means that the values of this nodes are scaled by 2**10 more than the child values.
/// The absolut offset is obtained by adding the offsets of all the nodes below this node (including)
pub fn forward_node<const DIM: usize>(
    child: usize,
    parent: usize,
    lin_partial_likelihoods: &mut [na::SVector<f64, DIM>],
    forward_data: &[ModelEdgeData<DIM>],
    param: &ParamPrecomp<DIM>,
    offsets: &mut [u32],
) {
    // Felsensteins rule
    // lin_partial_likelihoods[parent][a] *= sum_b lin_partial_likelihoods[child][b] * P(a -> b)
    // This is a matrix vector mutliplication Tv, where T is the transition matrix and v is the vector of partial likelihoods of the child node
    // T = V_pi * diag(exp_t_lambda) * V_pi_inv

    // All steps are quadratic, so no matrix matrix multiplication.
    let mul1 = param.V_pi_inv * lin_partial_likelihoods[child];
    let mul2 = forward_data[child].exp_t_lambda.component_mul(&mul1);
    let parent_contribution = param.V_pi * mul2;

    let mut max = 0.0;
    for a in 0..DIM {
        lin_partial_likelihoods[parent][a] *= parent_contribution[a];
        max = f64::max(max, lin_partial_likelihoods[parent][a]);
    }
    if max < f64::powi(2.0, -100) {
        for a in 0..DIM {
            lin_partial_likelihoods[parent][a] *= f64::powi(2.0, 100);
        }
        offsets[parent] += 100;
    }
}
