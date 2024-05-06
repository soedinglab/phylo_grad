use std::error::Error;

use crate::logsumexp::LogSumExp;

use crate::data_types::*;
use crate::tree::*;

impl FelsensteinError {
    pub const LEAF: Self = Self::LogicError("forward_node called on a leaf");
}

fn log_transition<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    t: Float,
) -> na::SMatrix<Float, DIM, DIM>
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let argument = rate_matrix * t;
    let matrix_exp = argument.exp();
    matrix_exp.map(Float::ln)
}

fn _child_input<const DIM: usize>(
    log_p: &[Float; DIM],
    distance: Float,
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM]
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(b, a) ) */
    let log_transition = log_transition(rate_matrix, distance);
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
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> Result<[Float; DIM], Box<dyn Error>>
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let node = &tree[id];
    /* TODO duplicate code */
    match (node.left, node.right) {
        (Some(left), Some(right)) => {
            let log_p_left = log_p[left].unwrap();
            let child_input_left = _child_input(&log_p_left, tree[left].distance, rate_matrix);

            let log_p_right = log_p[right].unwrap();
            let child_input_right = _child_input(&log_p_right, tree[right].distance, rate_matrix);

            let mut result = [0.0 as Float; DIM];
            for a in (0..DIM) {
                result[a] = child_input_left[a] + child_input_right[a];
            }
            Ok(result)
        }
        (Some(left), None) => {
            let log_p_left = log_p[left].unwrap();
            let result = _child_input(&log_p_left, tree[left].distance, rate_matrix);
            Ok(result)
        }
        (None, Some(right)) => {
            let log_p_right = log_p[right].unwrap();
            let result = _child_input(&log_p_right, tree[right].distance, rate_matrix);
            Ok(result)
        }
        (None, None) => Err(Box::new(FelsensteinError::LEAF)),
    }
}

pub fn forward_root<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<[Float; DIM]>],
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM]
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
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
            na::SVector::<Float, DIM>::from(_child_input(
                &log_p_child,
                tree[child].distance,
                rate_matrix,
            ))
        })
        .sum();
    <[Float; DIM]>::from(result)
}
