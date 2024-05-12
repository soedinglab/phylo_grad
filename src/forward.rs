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

pub fn forward_data_precompute<const DIM: usize>(
    forward_data: &mut Vec<LogTransitionForwardData<DIM>>,
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    distances: &[Float],
) where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    forward_data.clear();
    forward_data.extend(
        distances
            .iter()
            .map(|dist| log_transition_precompute(rate_matrix, *dist)),
    );
}

fn log_transition<const DIM: usize>(
    id: Id,
    forward_data: &[LogTransitionForwardData<DIM>],
) -> na::SMatrix<Float, DIM, DIM> {
    forward_data[id].step_2.map(Float::ln)
}

fn child_input<const DIM: usize>(
    child_id: Id, //only used in forward_data
    log_p: &[Float; DIM],
    forward_data: &[LogTransitionForwardData<DIM>],
) -> [Float; DIM] {
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(b, a) ) */
    let log_transition = log_transition(child_id, forward_data);
    /* TODO! Make sure indices are not flipped */
    let mut col_a;
    let mut result = [0.0 as Float; DIM];
    for a in (0..DIM) {
        col_a = log_transition.column(a);
        result[a] = (0..DIM).map(|b| (log_p[b] + col_a[b])).ln_sum_exp()
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

    let mut opt_running_sum: Option<[Float; DIM]> = None;
    for opt_child in [node.left, node.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, &log_p[child].unwrap(), forward_data);
            match opt_running_sum {
                Some(ref mut result) => {
                    for a in (0..DIM) {
                        result[a] += child_input[a];
                    }
                }
                None => opt_running_sum = Some(child_input),
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
    log_p: &[Option<[Float; DIM]>],
    //rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    forward_data: &[LogTransitionForwardData<DIM>],
) -> [Float; DIM] {
    let root = &tree[id];

    let mut result = child_input(root.parent, &log_p[root.parent].unwrap(), forward_data);
    for opt_child in [root.left, root.right] {
        if let Some(child) = opt_child {
            let child_input = child_input(child, &log_p[child].unwrap(), forward_data);
            for i in (0..DIM) {
                result[i] += child_input[i];
            }
        }
    }

    result
}
