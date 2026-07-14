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
    /// matrix_exp transposed
    pub transition_T: na::SMatrix<f64, DIM, DIM>,
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
    pub Q : na::SMatrix<f64, DIM, DIM>,
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
        rate_matrix[(i, i)] = -rate_matrix.row(i).sum();
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

    let mut matrix_exp = param.V_pi.clone_owned();
    times_diag_assign(matrix_exp.as_view_mut(), exp_t_lambda.iter().copied());
    matrix_exp *= param.V_pi_inv;

    matrix_exp.apply(|x| *x = f64::max(*x, f64::MIN_POSITIVE));
    ModelEdgeData {
        transition_T: matrix_exp.transpose(),
        exp_t_lambda,
    }
}

pub fn forward_data_precompute_param<const DIM: usize>(
    param: &ParamPrecomp<DIM>,
    distances: &[f64],
) -> ForwardData<DIM> {
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<DIM>::with_capacity(num_nodes);

    forward_data.model_edge_data.extend(
        distances
            .iter()
            .map(|dist| precompute_model_edge_data(param, *dist)),
    );
    forward_data
}

/// adds the log_p of the children to the log_p of the parent
/// Main part of the Felsenstein in Forward
/// log_p are the partial log likelihoods, they start with the leave nodes initialized. This function takes 2 computed log_p vectors
/// and writes the compbined result in the parent log_p vector
/// Offsets are scaling factors to prevent underflow. offsets[i] = 10 means that the values of this nodes are scaled by 2**10 more than the child values.
/// The absolut offset is obtained by adding the offsets of all the nodes below this node (including)
pub fn forward_node<const DIM: usize>(
    child: usize,
    parent: usize,
    lin_partial_likelihoods: &mut [na::SVector<f64, DIM>],
    forward_data: &ForwardData<DIM>,
    offsets: &mut [u32],
) {
    // In linspace log_p[parent]_a = sum_b (log_p[child](b) * transiton(rate_matrix, distance)(a,b) )
    let mut max = 0.0;
    for a in 0..DIM {
        let mut sum = 0.0;
        for b in 0..DIM {
            sum += lin_partial_likelihoods[child][b]
                * forward_data.model_edge_data[child].transition_T[(b, a)];
        }
        lin_partial_likelihoods[parent][a] *= sum;
        max = f64::max(max, sum);
    }
    if max < f64::powi(2.0, -100) {
        for a in 0..DIM {
            lin_partial_likelihoods[parent][a] *= f64::powi(2.0, 100);
        }
        offsets[parent] += 100;
    }
}
