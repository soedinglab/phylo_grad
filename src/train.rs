use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::backward::*;
use crate::data_types::*;
use crate::forward::*;
use crate::tree::*;

fn forward_column<F: FloatTrait, const DIM: usize>(
    mut leaf_log_p: Vec<na::SVector<F, DIM>>,
    tree: &[TreeNode],
    forward_data: &ForwardData<F, DIM>,
) -> Vec<na::SVector<F, DIM>> {
    let num_nodes = tree.len();
    let num_leaves = leaf_log_p.len();

    /* TODO remove copy */
    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, tree, &leaf_log_p, forward_data).unwrap();
        leaf_log_p.push(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, tree, &leaf_log_p, forward_data);
    leaf_log_p.push(log_p_root);
    leaf_log_p
}

fn process_likelihood<F: FloatTrait, const DIM: usize>(
    log_p_root: na::SVectorView<F, DIM>,
    log_p_prior: na::SVectorView<F, DIM>,
) -> (F, na::SVector<F, DIM>) {
    let lse_arg = log_p_root + log_p_prior;
    let log_likelihood_column = F::logsumexp(lse_arg.iter());
    let grad_log_p_outgoing = softmax(lse_arg.as_view());
    (log_likelihood_column, grad_log_p_outgoing)
}

fn d_rate_column_param<F: FloatTrait, const DIM: usize>(
    grad_log_p_root: na::SVectorView<F, DIM>,
    tree: &[TreeNode],
    log_p: &[na::SVector<F, DIM>],
    distances: &[F],
    forward_data: &ForwardData<F, DIM>,
    param: &ParamPrecomp<F, DIM>,
    num_leaves: usize,
) -> na::SMatrix<F, DIM, DIM> {
    /* Notice that child_input values are always added, so the log_p input for children is always the same.
    We will therefore store their common grad_log_p in the parent node's BackwardData. */
    /* TODO: it is possible to free grad_log_p's for the previous tree level. */
    let num_nodes = tree.len();
    let mut backward_data = Vec::<BackwardData<F, DIM>>::with_capacity(num_nodes - num_leaves);
    let mut grad_rate_column = na::SMatrix::<F, DIM, DIM>::zeros();
    /* root.backward */
    backward_data.push(BackwardData {
        grad_log_p: grad_log_p_root.clone_owned(),
    });
    /* node.backward for non-terminal nodes */
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = tree[id].parent;
        let parent_backward_id = num_nodes - parent_id - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let distance_current = distances[id];
        let fwd_data_current = &forward_data.log_transition[id];
        let (grad_rate, grad_log_p) = d_child_input_param(
            grad_log_p_input,
            distance_current,
            param,
            log_p_input,
            fwd_data_current,
            true,
        );
        grad_rate_column += grad_rate;
        backward_data.push(BackwardData {
            grad_log_p: grad_log_p.unwrap(),
        });
    }
    /* For leaves, we only compute grad_rate */
    for id in (0..num_leaves).rev() {
        let parent_id = &tree[id].parent;
        let parent_backward_id = num_nodes - parent_id - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let distance_current = distances[id];
        let fwd_data_current = &forward_data.log_transition[id];
        let (grad_rate, _) = d_child_input_param(
            grad_log_p_input,
            distance_current,
            param,
            log_p_input,
            fwd_data_current,
            false,
        );
        grad_rate_column += grad_rate;
    }
    grad_rate_column
}

struct TrainColumnParamResult<F, const DIM: usize> {
    log_likelihood: F,
    grad_delta: na::SMatrix<F, DIM, DIM>,
    grad_sqrt_pi: na::SVector<F, DIM>,
}

fn train_column_param<F: FloatTrait, const DIM: usize>(
    leaf_log_p: Vec<na::SVector<F, DIM>>,
    S: na::SMatrixView<F, DIM, DIM>,
    sqrt_pi: na::SVectorView<F, DIM>,
    tree: &[TreeNode],
    distances: &[F],
) -> TrainColumnParamResult<F, DIM> {
    let num_leaves = leaf_log_p.len();
    // If the diagonalization fails or eigenvalues are to big, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S, sqrt_pi) {
        Some(param) => param,
        None => {
            return TrainColumnParamResult::<F, DIM> {
                log_likelihood: <F as num_traits::Float>::neg_infinity(),
                grad_delta: na::SMatrix::<F, DIM, DIM>::zeros(),
                grad_sqrt_pi: na::SVector::<F, DIM>::zeros(),
            }
        }
    };

    let forward_data = forward_data_precompute_param(&param, distances);

    let log_p = forward_column(leaf_log_p, tree, &forward_data);
    let log_p_root = log_p.last().unwrap();

    let log_p_prior = sqrt_pi.map(num_traits::Float::ln) * <F as FloatTrait>::from_f64(2.0);
    let (log_likelihood, grad_log_p_likelihood): (F, na::SVector<F, DIM>) =
        process_likelihood(log_p_root.as_view(), log_p_prior.as_view());
    let grad_log_prior = grad_log_p_likelihood;
    let grad_log_p_root = grad_log_p_likelihood;

    let grad_rate = d_rate_column_param(
        grad_log_p_root.as_view(),
        tree,
        &log_p,
        distances,
        &forward_data,
        &param,
        num_leaves,
    );

    let (grad_delta, mut grad_sqrt_pi) = d_param(grad_rate.as_view(), &param);

    let mut grad_sqrt_pi_likelihood: na::SMatrix<F, DIM, 1> =
        param.sqrt_pi_recip * <F as FloatTrait>::from_f64(2.0);
    grad_sqrt_pi_likelihood.component_mul_assign(&grad_log_prior);
    grad_sqrt_pi += grad_sqrt_pi_likelihood;
    TrainColumnParamResult::<F, DIM> {
        log_likelihood,
        grad_delta,
        grad_sqrt_pi,
    }
}

pub struct InferenceResultParam<F, const DIM: usize> {
    pub log_likelihood_total: Vec<F>,
    pub grad_delta_total: Vec<na::SMatrix<F, DIM, DIM>>,
    pub grad_sqrt_pi_total: Vec<na::SVector<F, DIM>>,
}

pub fn train_parallel_param_unpaired<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &[Vec<na::SVector<F, DIM>>],
    S: &[na::SMatrix<F, DIM, DIM>],
    sqrt_pi: &[na::SVector<F, DIM>],
    tree: &[TreeNode],
    distances: &[F],
) -> InferenceResultParam<F, DIM> {
    let col_results = (leaf_log_p, S, sqrt_pi)
        .into_par_iter()
        .map(|(leaf_log_p, S, sqrt_pi)| {
            train_column_param(
                leaf_log_p.clone(),
                S.as_view(),
                sqrt_pi.as_view(),
                tree,
                distances,
            )
        })
        .collect::<Vec<_>>();

    let mut log_likelihood_total = vec![];
    let mut grad_delta_total = vec![];
    let mut grad_sqrt_pi_total = vec![];

    for col_result in col_results {
        log_likelihood_total.push(col_result.log_likelihood);
        grad_delta_total.push(col_result.grad_delta);
        grad_sqrt_pi_total.push(col_result.grad_sqrt_pi);
    }

    InferenceResultParam {
        log_likelihood_total,
        grad_delta_total,
        grad_sqrt_pi_total,
    }
}
