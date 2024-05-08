use std::error::Error;

use crate::logsumexp::LogSumExp;

use crate::data_types::*;
use crate::tree::*;

impl FelsensteinError {
    pub const LEAF: Self = Self::LogicError("forward_node called on a leaf");
}

pub struct LogTransitionForwardData<const DIM: usize> {
    pub step_1: na::SMatrix<Float, DIM, DIM>,
    pub step_2: na::SMatrix<Float, DIM, DIM>,
}

fn log_transition_precompute<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distance: Float,
) -> LogTransitionForwardData<DIM>
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let step_1 = rate_matrix * distance;
    let step_2 = step_1.exp();
    //let result = step_2.map(Float::ln);

    LogTransitionForwardData { step_1, step_2 }
}

pub fn forward_data_precompute<const DIM: usize, I>(
    forward_data: &mut Vec<LogTransitionForwardData<DIM>>,
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distances: &[Float],
    ids: I,
) where
    I: IntoIterator<Item = Id>,
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    forward_data.clear();
    forward_data.extend(
        ids.into_iter()
            .map(|id| log_transition_precompute(rate_matrix, distances[id])),
    );
}

fn log_transition<const DIM: usize>(
    id: Id,
    forward_data: &[LogTransitionForwardData<DIM>],
) -> na::SMatrix<Float, DIM, DIM> {
    forward_data[id].step_2.map(Float::ln)
}

fn _child_input<const DIM: usize>(
    child_id: Id, //only used in forward_data
    log_p: &[Float; DIM],
    forward_data: &[LogTransitionForwardData<DIM>],
) -> [Float; DIM] {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(b, a) ) */
    let log_transition = log_transition(child_id, forward_data);
    /* Is this better or worse than adding two nalgebra vectors and taking logsumexp?
    What is the optimal way to access a matrix? */
    let mut result = [0.0 as Float; DIM];
    for a in (0..DIM) {
        result[a] = (0..DIM)
            .map(|b| (log_p[b] + log_transition[(a, b)]))
            .ln_sum_exp()
    }
    result
}

/// forward_node expects that the node tree[id] is non-terminal!
/// To initialize a leaf node, call Entry::to_log_p().
pub fn forward_node<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<[Float; DIM]>],
    //rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    forward_data: &[LogTransitionForwardData<DIM>],
) -> Result<[Float; DIM], FelsensteinError> {
    let node = &tree[id];
    /* TODO duplicate code */
    match (node.left, node.right) {
        (Some(left), Some(right)) => {
            let log_p_left = log_p[left].unwrap();
            let child_input_left = _child_input(left, &log_p_left, forward_data);

            let log_p_right = log_p[right].unwrap();
            let child_input_right = _child_input(right, &log_p_right, forward_data);

            let mut result = [0.0 as Float; DIM];
            for a in (0..DIM) {
                result[a] = child_input_left[a] + child_input_right[a];
            }
            Ok(result)
        }
        (Some(left), None) => {
            let log_p_left = log_p[left].unwrap();
            let result = _child_input(left, &log_p_left, forward_data);
            Ok(result)
        }
        (None, Some(right)) => {
            let log_p_right = log_p[right].unwrap();
            let result = _child_input(right, &log_p_right, forward_data);
            Ok(result)
        }
        (None, None) => Err(FelsensteinError::LEAF),
    }
}

pub fn forward_root<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<[Float; DIM]>],
    //rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    forward_data: &[LogTransitionForwardData<DIM>],
) -> [Float; DIM] {
    let root = &tree[id];

    let mut children = Vec::with_capacity(3);
    children.push(root.parent);
    if let Some(child) = root.left {
        children.push(child);
    }
    if let Some(child) = root.right {
        children.push(child);
    }

    let result: na::SVector<Float, DIM> = children
        .into_iter()
        .map(|child| {
            let log_p_child = log_p[child].unwrap();
            na::SVector::<Float, DIM>::from(_child_input(child, &log_p_child, forward_data))
        })
        .sum();
    <[Float; DIM]>::from(result)
}
