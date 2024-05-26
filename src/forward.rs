use crate::data_types::*;
use crate::tree::*;
use logsumexp::LogSumExp;

use na::{Const, DefaultAllocator};

impl FelsensteinError {
    pub const LEAF: Self = Self::LogicError("forward_node called on a leaf");
}

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

fn log_transition_precompute_symmetric<const DIM: usize>(
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
    log_p: &[Float; DIM],
    forward_data: &ForwardData<DIM>,
) -> [Float; DIM] {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(a, b) ) */
    let log_transition = &forward_data.log_transition[child_id].log_transition();

    let mut row_a;
    let mut result = [0.0 as Float; DIM];
    for a in 0..DIM {
        row_a = log_transition.row(a);
        result[a] = (0..DIM).map(|b| (log_p[b] + row_a[b])).ln_sum_exp()
    }
    result
}

/// forward_node expects that the node tree[id] is non-terminal!
/// To initialize a leaf node, call Entry::to_log_p().
pub fn forward_node<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[[Float; DIM]],
    forward_data: &ForwardData<DIM>,
) -> Result<[Float; DIM], FelsensteinError> {
    let node = &tree[id];

    let mut opt_running_sum: Option<[Float; DIM]> = None;
    for opt_child in [node.left, node.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, &log_p[child], forward_data);
            match opt_running_sum {
                Some(ref mut result) => {
                    for a in 0..DIM {
                        result[a] += child_input[a];
                    }
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
    log_p: &[[Float; DIM]],
    forward_data: &ForwardData<DIM>,
) -> [Float; DIM] {
    let root = &tree[id];

    let mut result = child_input(root.parent, &log_p[root.parent], forward_data);
    for opt_child in [root.left, root.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, &log_p[child], forward_data);
            for i in 0..DIM {
                result[i] += child_input[i];
            }
        }
    }

    result
}
