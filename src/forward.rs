use crate::data_types::*;
use crate::tree::*;

use nalgebra_lapack::SymmetricEigen;

pub struct ForwardData<F, const DIM: usize> {
    pub log_transition: Vec<LogTransitionForwardData<F, DIM>>,
}

impl<F, const DIM: usize> ForwardData<F, DIM> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            log_transition: Vec::with_capacity(capacity),
        }
    }
}

pub struct LogTransitionForwardData<F, const DIM: usize> {
    pub matrix_exp: na::SMatrix<F, DIM, DIM>,
    pub log_transition: na::SMatrix<F, DIM, DIM>,
}

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

    let sqrt_pi_recip = sqrt_pi.map(|x| Float::recip(Float::max(x, F::EPS_DIV)));

    let mut S_symmetric = S.clone_owned();
    for i in 0..DIM {
        for j in 0..i {
            S_symmetric[(i, j)] = S_symmetric[(j, i)];
        }
    }

    /* TODO remove */
    /* rate_matrix = diag(sqrt_pi_recip) * S_output * diag(sqrt_pi) */
    let mut rate_matrix = S_symmetric.clone_owned();
    diag_times_assign(rate_matrix.as_view_mut(), sqrt_pi_recip.iter().copied());
    times_diag_assign(rate_matrix.as_view_mut(), sqrt_pi.iter().copied());

    for i in 0..DIM {
        S_symmetric[(i, i)] = -rate_matrix.row(i).sum() + rate_matrix[(i, i)];
    }

    let SymmetricEigen {
        eigenvalues,
        eigenvectors,
    } = nalgebra_lapack::SymmetricEigen::try_new(S_symmetric)?;

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

    let mut matrix_exp = param.V_pi.clone_owned();
    times_diag_assign(
        matrix_exp.as_view_mut(),
        param
            .eigenvalues
            .iter()
            .map(|lam| Float::exp(*lam * distance)),
    );
    matrix_exp *= param.V_pi_inv;

    let log_transition = matrix_exp.map(|x| Float::ln(Float::max(x, F::EPS_LOG)));

    LogTransitionForwardData {
        matrix_exp,
        log_transition,
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

/// Computes p(subtree | parent = a) for all a by taking log_p : p(subtree | child = b) for all b
fn child_input<F: FloatTrait, const DIM: usize>(
    log_p: na::SVectorView<F, DIM>,
    log_transition: &LogTransitionForwardData<F, DIM>,
) -> na::SVector<F, DIM> {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(a, b) ) */
    let log_transition = log_transition.log_transition;

    let mut result = na::SVector::<F, DIM>::zeros();
    for a in 0..DIM {
        let row_a = log_transition.row(a).transpose();
        result[a] = F::logsumexp((log_p + row_a).iter());
    }
    result
}

/// forward_node expects that the node tree[id] is non-terminal!
/// To initialize a leaf node, call Entry::to_log_p().
pub fn forward_node<F: FloatTrait, const DIM: usize>(
    id: usize,
    tree: &[TreeNode],
    log_p: &[na::SVector<F, DIM>],
    forward_data: &ForwardData<F, DIM>,
) -> na::SVector<F, DIM> {
    let node = &tree[id];

    let mut opt_running_sum: Option<na::SVector<F, DIM>> = None;
    for child in [node.left, node.right].into_iter().flatten() {
        let child_input = child_input(log_p[child as usize].as_view(), &forward_data.log_transition[child as usize]);
        match opt_running_sum {
            Some(ref mut result) => {
                *result += child_input;
            }
            None => {
                opt_running_sum = Some(child_input);
            }
        }
    }

    match opt_running_sum {
        Some(result) => result,
        None => panic!("Non-terminal node without children"),
    }
}

pub fn forward_root<F: FloatTrait, const DIM: usize>(
    id: usize,
    tree: &[TreeNode],
    log_p: &[na::SVector<F, DIM>],
    forward_data: &ForwardData<F, DIM>,
) -> na::SVector<F, DIM> {
    let root = &tree[id];

    let mut result = child_input(
        //root uses the parent field to store the third child
        log_p[root.parent as usize].as_view(),
        &forward_data.log_transition[root.parent as usize],
    );
    for opt_child in [root.left, root.right] {
        if let Some(child) = opt_child {
            let child_input =
                child_input(log_p[child as usize].as_view(), &forward_data.log_transition[child as usize]);
            result += child_input;
        }
    }

    result
}
