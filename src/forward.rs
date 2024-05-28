use crate::data_types::*;
use crate::tree::*;
use logsumexp::LogSumExp;

use na::{Const, DefaultAllocator};

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
    pub step_1: na::SMatrix<Float, DIM, DIM>,
    pub step_2: na::SMatrix<Float, DIM, DIM>,
}
impl<const DIM: usize> LogTransitionForwardData<DIM> {
    pub fn log_transition(&self) -> na::SMatrix<Float, DIM, DIM> {
        self.step_2.map(Float::ln)
    }
}
pub struct ParamData<const DIM: usize> {
    pub symmetric_matrix: na::SMatrix<Float, DIM, DIM>,
    pub sqrt_pi: na::SVector<Float, DIM>,
    pub sqrt_pi_recip: na::SVector<Float, DIM>,
    pub eigenvectors: na::SMatrix<Float, DIM, DIM>,
    pub eigenvalues: na::SVector<Float, DIM>,
    pub V_pi: na::SMatrix<Float, DIM, DIM>,
    pub V_pi_inv: na::SMatrix<Float, DIM, DIM>,
    pub rate_matrix: na::SMatrix<Float, DIM, DIM>,
}

/// In-place multiplication by a diagonal matrix on the left
pub fn diag_times_assign<I, const N: usize>(
    mut matrix: na::SMatrixViewMut<Float, N, N>,
    diagonal_entries: I,
) where
    I: Iterator<Item = Float>,
{
    for (mut row, scale) in std::iter::zip(matrix.row_iter_mut(), diagonal_entries) {
        row *= scale;
    }
}

/// In-place multiplication by a diagonal matrix on the right
pub fn times_diag_assign<I, const N: usize>(
    mut matrix: na::SMatrixViewMut<Float, N, N>,
    diagonal_entries: I,
) where
    I: Iterator<Item = Float>,
{
    for (mut col, scale) in std::iter::zip(matrix.column_iter_mut(), diagonal_entries) {
        col *= scale;
    }
}

pub fn compute_param_data<const DIM: usize>(
    delta: na::SMatrixView<Float, DIM, DIM>,
    sqrt_pi: na::SVectorView<Float, DIM>,
) -> ParamData<DIM>
where
    Const<DIM>: Decrementable,
    DefaultAllocator: DecrementableAllocator<Float, DIM>,
{
    let sqrt_pi_recip = sqrt_pi.map(Float::recip);

    let mut symmetric_output = delta.clone_owned();
    for i in 0..DIM {
        for j in 0..i {
            symmetric_output[(i, j)] = symmetric_output[(j, i)];
        }
    }

    /* TODO remove */
    /* rate_matrix = diag(sqrt_pi_recip) * S_output * diag(sqrt_pi) */
    let mut rate_matrix = symmetric_output.clone_owned();
    diag_times_assign(rate_matrix.as_view_mut(), sqrt_pi_recip.iter().copied());
    times_diag_assign(rate_matrix.as_view_mut(), sqrt_pi.iter().copied());

    for i in 0..DIM {
        let row = rate_matrix.row(i).clone_owned();
        symmetric_output[(i, i)] = -(row.as_slice()[..i].iter().sum::<Float>()
            + row.as_slice()[i + 1..].iter().sum::<Float>());
        rate_matrix[(i, i)] = symmetric_output[(i, i)];
    }

    let na::SymmetricEigen {
        eigenvalues,
        eigenvectors,
    } = symmetric_output.symmetric_eigen();

    let mut V_pi = eigenvectors.clone_owned();
    diag_times_assign(V_pi.as_view_mut(), sqrt_pi_recip.iter().copied());

    let mut V_pi_inv = eigenvectors.transpose();
    times_diag_assign(V_pi_inv.as_view_mut(), sqrt_pi.iter().copied());

    ParamData {
        symmetric_matrix: symmetric_output,
        sqrt_pi: sqrt_pi.clone_owned(),
        sqrt_pi_recip,
        eigenvectors,
        eigenvalues,
        V_pi,
        V_pi_inv,
        rate_matrix,
    }
}

fn log_transition_precompute_param<const DIM: usize>(
    param: &ParamData<DIM>,
    distance: Float,
) -> LogTransitionForwardData<DIM> {
    /* TODO remove step_1 */
    let step_1 = param.rate_matrix * distance;

    let mut step_2 = param.V_pi.clone_owned();
    times_diag_assign(
        step_2.as_view_mut(),
        param.eigenvalues.iter().map(|lam| (lam * distance).exp()),
    );
    step_2 *= param.V_pi_inv;

    LogTransitionForwardData { step_1, step_2 }
}

pub fn forward_data_precompute_param<const DIM: usize>(
    param: &ParamData<DIM>,
    distances: &[Float],
) -> ForwardData<DIM>
where
    Const<DIM>: Decrementable,
    DefaultAllocator: DecrementableAllocator<Float, DIM>,
{
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<DIM>::with_capacity(num_nodes);

    forward_data.log_transition.extend(
        distances
            .iter()
            .map(|dist| log_transition_precompute_param(&param, *dist)),
    );
    forward_data
}

/* fn log_transition_precompute_symmetric<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    rate_symmetric_eigen: &na::SymmetricEigen<Float, Const<DIM>>,
    distance: Float,
) -> LogTransitionForwardData<DIM> {
    let step_1 = rate_matrix * distance;
    let new_eigenvalues = (rate_symmetric_eigen.eigenvalues * distance).map(Float::exp);
    let step_2 = na::SymmetricEigen {
        eigenvectors: rate_symmetric_eigen.eigenvectors,
        eigenvalues: new_eigenvalues,
    }
    .recompose();
    LogTransitionForwardData { step_1, step_2 }
}

pub fn forward_data_precompute_symmetric<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distances: &[Float],
) -> ForwardData<DIM>
where
    Const<DIM>: Decrementable,
    DefaultAllocator: DecrementableAllocator<Float, DIM>,
{
    let num_nodes = distances.len();
    let mut forward_data = ForwardData::<DIM>::with_capacity(num_nodes);
    /* What does try_symmetric_eigen() do? */
    let rate_symmetric_eigen = rate_matrix.symmetric_eigen();

    forward_data
        .log_transition
        .extend(distances.iter().map(|dist| {
            log_transition_precompute_symmetric(rate_matrix, &rate_symmetric_eigen, *dist)
        }));
    forward_data
} */

fn log_transition_precompute<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
) -> LogTransitionForwardData<DIM>
where
    na::Const<DIM>: Exponentiable,
{
    let step_1 = rate_matrix * distance;
    let step_2 = step_1.exp();

    LogTransitionForwardData { step_1, step_2 }
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
    child_id: Id, //only used in forward_data
    log_p: na::SVectorView<Float, DIM>,
    forward_data: &ForwardData<DIM>,
) -> na::SVector<Float, DIM> {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(a, b) ) */
    let log_transition = &forward_data.log_transition[child_id].log_transition();

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
    id: Id,
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
    id: Id,
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
