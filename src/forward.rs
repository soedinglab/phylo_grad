use crate::data_types::*;
use crate::tree::*;
use logsumexp::LogSumExp;

use na::Const;
use nalgebra_lapack::SymmetricEigen;

impl FelsensteinError {
    pub const LEAF: Self = Self::LogicError("forward_node called on a leaf");
}

/* TODO ForwardDataParam */

pub struct ForwardData<const DIM: usize> {
    pub log_transition: Vec<LogTransitionForwardData<DIM>>,
}

impl<const DIM: usize> ForwardData<DIM> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            log_transition: Vec::with_capacity(capacity),
        }
    }
}

pub struct LogTransitionForwardData<const DIM: usize> {
    pub step_1: Option<na::SMatrix<Float, DIM, DIM>>,
    pub step_2: na::SMatrix<Float, DIM, DIM>,
    pub log_transition: na::SMatrix<Float, DIM, DIM>,
}

pub struct ParamPrecomp<F, const DIM: usize> {
    pub symmetric_matrix: na::SMatrix<F, DIM, DIM>,
    pub sqrt_pi: na::SVector<F, DIM>,
    pub sqrt_pi_recip: na::SVector<F, DIM>,
    pub eigenvectors: na::SMatrix<F, DIM, DIM>,
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

pub fn compute_param_data<const DIM: usize>(
    S: na::SMatrixView<Float, DIM, DIM>,
    sqrt_pi: na::SVectorView<Float, DIM>,
) -> Option<ParamPrecomp<Float, DIM>> {
    let sqrt_pi_recip = sqrt_pi.map(|x| Float::recip(x.max(EPS_DIV as Float)));

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
        let row = rate_matrix.row(i).clone_owned();
        S_symmetric[(i, i)] = -(row.as_slice()[..i].iter().sum::<Float>()
            + row.as_slice()[i + 1..].iter().sum::<Float>());
        rate_matrix[(i, i)] = S_symmetric[(i, i)];
    }

    let SymmetricEigen{eigenvalues, eigenvectors} = nalgebra_lapack::SymmetricEigen::try_new(S_symmetric)?;

    // Prevent numerical instability
    let norm_eigenvals = eigenvalues.iter().map(|x| x.abs()).sum::<Float>();
    if norm_eigenvals > 1e5 {
        return None;
    }

    let mut V_pi = eigenvectors.clone_owned();
    diag_times_assign(V_pi.as_view_mut(), sqrt_pi_recip.iter().copied());

    let mut V_pi_inv = eigenvectors.transpose();
    times_diag_assign(V_pi_inv.as_view_mut(), sqrt_pi.iter().copied());

    Some(ParamPrecomp {
        symmetric_matrix: S_symmetric,
        sqrt_pi: sqrt_pi.clone_owned(),
        sqrt_pi_recip,
        eigenvectors,
        eigenvalues,
        V_pi,
        V_pi_inv,
    })
}

fn log_transition_precompute_param<const DIM: usize>(
    param: &ParamPrecomp<Float, DIM>,
    distance: Float,
) -> LogTransitionForwardData<DIM> {
    let mut step_2 = param.V_pi.clone_owned();
    times_diag_assign(
        step_2.as_view_mut(),
        param.eigenvalues.iter().map(|lam| (lam * distance).exp()),
    );
    step_2 *= param.V_pi_inv;

    let log_transition = step_2.map(|x| Float::ln(x.max(EPS_LOG as Float)));

    LogTransitionForwardData {
        step_1: None,
        step_2,
        log_transition,
    }
}

pub fn forward_data_precompute_param<const DIM: usize>(
    param: &ParamPrecomp<Float, DIM>,
    distances: &[Float],
) -> ForwardData<DIM> {
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<DIM>::with_capacity(num_nodes);

    forward_data.log_transition.extend(
        distances
            .iter()
            .map(|dist| log_transition_precompute_param(param, *dist)),
    );
    forward_data
}

fn log_transition_precompute<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
) -> LogTransitionForwardData<DIM>
where
    na::Const<DIM>: Exponentiable,
{
    let step_1 = rate_matrix * distance;
    let step_2 = step_1.exp();
    let log_transition = step_2.map(|x| Float::ln(x.max(EPS_LOG as Float)));

    LogTransitionForwardData {
        step_1: Some(step_1),
        step_2,
        log_transition,
    }
}

pub fn forward_data_precompute<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distances: &[Float],
) -> ForwardData<DIM>
where
    Const<DIM>: Exponentiable,
{
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<DIM>::with_capacity(num_nodes);

    forward_data.log_transition.extend(
        distances
            .iter()
            .map(|dist| log_transition_precompute(rate_matrix, *dist)),
    );
    forward_data
}

fn child_input<const DIM: usize>(
    child_id: usize, //only used in forward_data
    log_p: na::SVectorView<Float, DIM>,
    forward_data: &ForwardData<DIM>,
) -> na::SVector<Float, DIM> {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(a, b) ) */
    let log_transition = forward_data.log_transition[child_id].log_transition;

    let mut result = na::SVector::<Float, DIM>::zeros();
    for a in 0..DIM {
        let row_a = log_transition.row(a).transpose();
        result[a] = (log_p + row_a).iter().ln_sum_exp();
    }
    result
}

/// forward_node expects that the node tree[id] is non-terminal!
/// To initialize a leaf node, call Entry::to_log_p().
pub fn forward_node<const DIM: usize>(
    id: usize,
    tree: &[TreeNode],
    log_p: &[na::SVector<Float, DIM>],
    forward_data: &ForwardData<DIM>,
) -> Result<na::SVector<Float, DIM>, FelsensteinError> {
    let node = &tree[id];

    let mut opt_running_sum: Option<na::SVector<Float, DIM>> = None;
    for opt_child in [node.left, node.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, log_p[child].as_view(), forward_data);
            match opt_running_sum {
                Some(ref mut result) => {
                    *result += child_input;
                }
                None => {
                    opt_running_sum = Some(child_input);
                }
            }
        }
    }

    match opt_running_sum {
        Some(result) => Ok(result),
        None => Err(FelsensteinError::LEAF),
    }
}

pub fn forward_root<const DIM: usize>(
    id: usize,
    tree: &[TreeNode],
    log_p: &[na::SVector<Float, DIM>],
    forward_data: &ForwardData<DIM>,
) -> na::SVector<Float, DIM> {
    let root = &tree[id];

    let mut result = child_input(root.parent, log_p[root.parent].as_view(), forward_data);
    for opt_child in [root.left, root.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, log_p[child].as_view(), forward_data);
            result += child_input;
        }
    }

    result
}
