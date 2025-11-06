use crate::data_types::*;

use nalgebra as na;

/// Forward data precomputed before the forward pass
pub struct ForwardData<F, const DIM: usize> {
    pub log_transition: Vec<LogTransitionForwardData<F, DIM>>,
}

/// Forward data which is saved during the forward pass
pub struct ForwardDataSave<F, const DIM: usize> {
    pub logsumexp_exp_save: Vec<na::SMatrix<F, DIM, DIM>>,
    pub logsumexp_sum_save: Vec<na::SVector<F, DIM>>,
}

impl<F : FloatTrait, const DIM: usize> ForwardDataSave<F, DIM> {
    pub fn new(capacity: usize) -> Self {
        Self {
            logsumexp_exp_save: vec![
                na::SMatrix::<F, DIM, DIM>::zeros();
                capacity
            ],
            logsumexp_sum_save: vec![
                na::SVector::<F, DIM>::zeros();
                capacity
            ],
        }
    }
}

impl<F, const DIM: usize> ForwardData<F, DIM> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            log_transition: Vec::with_capacity(capacity),
        }
    }
}

#[derive(Debug)]
pub struct LogTransitionForwardData<F, const DIM: usize> {
    pub matrix_exp_recip: na::SMatrix<F, DIM, DIM>,
    pub log_transition_T: na::SMatrix<F, DIM, DIM>,
    pub exp_t_lambda: na::SVector<F, DIM>,
}

#[derive(Debug)]
pub struct ParamPrecomp<F, const DIM: usize> {
    pub symmetric_matrix: na::SMatrix<F, DIM, DIM>,
    pub sqrt_pi: na::SVector<F, DIM>,
    pub sqrt_pi_recip: na::SVector<F, DIM>,
    pub eigenvalues: na::SVector<F, DIM>,
    pub V_pi: na::SMatrix<F, DIM, DIM>,
    pub V_pi_inv: na::SMatrix<F, DIM, DIM>,
}

/// In-place multiplication by a diagonal matrix on the left
pub fn diag_times_assign<I, F, const N: usize>(
    mut matrix: na::SMatrixViewMut<F, N, N>,
    diagonal_entries: I,
) where
    F: FloatTrait,
    I: Iterator<Item = F>,
{
    for (mut row, scale) in std::iter::zip(matrix.row_iter_mut(), diagonal_entries) {
        row *= scale;
    }
}

/// In-place multiplication by a diagonal matrix on the right
pub fn times_diag_assign<I, F, const N: usize>(
    mut matrix: na::SMatrixViewMut<F, N, N>,
    diagonal_entries: I,
) where
    F: FloatTrait,
    I: Iterator<Item = F>,
{
    for (mut col, scale) in std::iter::zip(matrix.column_iter_mut(), diagonal_entries) {
        col *= scale;
    }
}

/// Precomputes things out of S and sqrt_pi
/// Returns None if the eigenvalues are too large or the diagonalization failed, this can happen with extreme values
pub fn compute_param_data<F: FloatTrait, const DIM: usize>(
    S: na::SMatrixView<F, DIM, DIM>,
    sqrt_pi: na::SVectorView<F, DIM>,
) -> Option<ParamPrecomp<F, DIM>> {
    use num_traits::Float;

    let sqrt_pi_recip = sqrt_pi.map(|x| Float::recip(Float::max(x, F::MIN_SQRT_PI)));

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
        S_symmetric[(i, i)] = -rate_matrix.row(i).sum() + rate_matrix[(i, i)];
    }

    let (eigenvalues, eigenvectors) = F::symmetric_eigen(S_symmetric)?;

    // Prevent numerical instability
    let norm_eigenvals = eigenvalues.iter().map(|x| x.abs()).sum::<F>();
    if norm_eigenvals > <F as FloatTrait>::from_f64(1e5) {
        return None;
    }

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
    })
}

fn log_transition_precompute_param<F: FloatTrait, const DIM: usize>(
    param: &ParamPrecomp<F, DIM>,
    distance: F,
) -> LogTransitionForwardData<F, DIM> {
    use num_traits::Float;

    let exp_t_lambda = param.eigenvalues.map(|lam| Float::exp(lam * distance));

    let mut matrix_exp = param.V_pi.clone_owned();
    times_diag_assign(matrix_exp.as_view_mut(), exp_t_lambda.iter().copied());
    matrix_exp *= param.V_pi_inv;

    matrix_exp.apply(|x| *x = Float::max(*x, F::MIN_SQRT_PI));

    let log_transition = matrix_exp.map(Float::ln);

    matrix_exp.apply(|x| *x = Float::recip(*x));

    LogTransitionForwardData {
        matrix_exp_recip: matrix_exp,
        log_transition_T: log_transition.transpose(),
        exp_t_lambda,
    }
}

pub fn forward_data_precompute_param<F: FloatTrait, const DIM: usize>(
    param: &ParamPrecomp<F, DIM>,
    distances: &[F],
) -> ForwardData<F, DIM> {
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<F, DIM>::with_capacity(num_nodes);

    forward_data.log_transition.extend(
        distances
            .iter()
            .map(|dist| log_transition_precompute_param(param, *dist)),
    );
    forward_data
}

/// adds the log_p of the children to the log_p of the parent
pub fn forward_node<F: FloatTrait, const DIM: usize>(
    child: usize,
    parent: usize,
    log_p: &mut [na::SVector<F, DIM>],
    forward_data: &ForwardData<F, DIM>,
    forward_data_save: &mut ForwardDataSave<F, DIM>,
) {
    let logsumexp_exp_save = &mut forward_data_save.logsumexp_exp_save[child].data.0;
    let logsumexp_sum_save = forward_data_save.logsumexp_sum_save[child].as_mut_slice();
    /* log_p[parent]_a = logsumexp_b(log_p[child](b) + log_transition(rate_matrix, distance)(a, b) ) */
    for a in 0..DIM {
        let row_a = forward_data.log_transition[child].log_transition_T.column(a);
        let tmp = log_p[child] + row_a;
        unsafe {
            log_p[parent][a] += F::vec_logsumexp_save(std::mem::transmute::<&[[F; DIM]; 1], &[F; DIM]>(
                &tmp.data.0), &mut logsumexp_exp_save[a], &mut logsumexp_sum_save[a]);
        }
    }
}