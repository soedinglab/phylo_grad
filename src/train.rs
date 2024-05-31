use logsumexp::LogSumExp;
use na::{Const, DefaultAllocator};
use rayon::prelude::*;
//use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::backward::*;
use crate::data_types::*;
use crate::forward::*;
use crate::tree::*;

fn forward_column<const DIM: usize, Entry, D>(
    column_iter: impl Iterator<Item = Entry>,
    distribution: &D,
    tree: &[TreeNode],
    forward_data: &ForwardData<DIM>,
) -> Vec<na::SVector<Float, DIM>>
where
    Entry: EntryTrait,
    D: Distribution<Entry, Float, DIM>,
{
    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    let num_nodes = tree.len();
    /* --- Leaf initialization --- */
    let mut log_p = Vec::<na::SVector<Float, DIM>>::with_capacity(num_nodes);
    log_p.extend(column_iter.map(|x| distribution.log_p(x)));
    let num_leaves = log_p.len();
    /* --- / Leaf initialization --- */

    /* TODO remove copy */
    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, tree, &log_p, forward_data).unwrap();
        log_p.push(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, tree, &log_p, forward_data);
    log_p.push(log_p_root);
    log_p
}

fn process_likelihood<const DIM: usize>(
    log_p_root: na::SVectorView<Float, DIM>,
    log_p_prior: na::SVectorView<Float, DIM>,
) -> (Float, na::SVector<Float, DIM>) {
    let lse_arg = log_p_root + log_p_prior;
    let log_likelihood_column = lse_arg.iter().ln_sum_exp();
    let grad_log_p_outgoing = softmax_na(lse_arg.as_view());
    (log_likelihood_column, grad_log_p_outgoing)
}

fn d_rate_column<const DIM: usize>(
    grad_log_p_root: na::SVectorView<Float, DIM>,
    tree: &[TreeNode],
    log_p: &[na::SVector<Float, DIM>],
    distances: &[Float],
    forward_data: &ForwardData<DIM>,
    num_leaves: usize,
) -> na::SMatrix<Float, DIM, DIM>
where
    Const<DIM>: Doubleable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>,
{
    /* Notice that child_input values are always added, so the log_p input for children is always the same.
    We will therefore store their common grad_log_p in the parent node's BackwardData. */
    /* TODO: it is possible to free grad_log_p's for the previous tree level. */
    let num_nodes = tree.len();
    let mut backward_data = Vec::<BackwardData<DIM>>::with_capacity(num_nodes - num_leaves);
    let mut grad_rate_column = na::SMatrix::<Float, DIM, DIM>::from_element(0.0 as Float);
    /* root.backward */
    backward_data.push(BackwardData {
        grad_log_p: grad_log_p_root.clone_owned(),
    });
    /* node.backward for non-terminal nodes */
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = &tree[id].parent;
        let parent_backward_id = num_nodes - parent_id - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let distance_current = distances[id];
        let fwd_data_current = &forward_data.log_transition[id];
        let (grad_rate, grad_log_p) = d_child_input_vjp(
            grad_log_p_input,
            distance_current,
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
        let (grad_rate, _) = d_child_input_vjp(
            grad_log_p_input,
            distance_current,
            log_p_input,
            fwd_data_current,
            false,
        );
        grad_rate_column += grad_rate;
    }
    grad_rate_column
}

/* TODO should we let rayon know that index_pairs is a vector and not just any slice? */
/* TODO Id<DIM> */
pub fn train_parallel<const DIM: usize, Residue, D>(
    index_pairs: &[(usize, usize)],
    residue_sequences_2d: na::DMatrixView<Residue>,
    rate_matrices: &[na::SMatrix<Float, DIM, DIM>],
    log_p_priors: &[na::SVector<Float, DIM>],
    tree: &[TreeNode],
    distances: &[Float],
    distribution: &D,
) -> (
    Vec<Float>,
    Vec<na::SMatrix<Float, DIM, DIM>>,
    Vec<na::SVector<Float, DIM>>,
)
where
    Residue: ResidueTrait,
    D: Distribution<ResiduePair<Residue>, Float, DIM>,
    //ResiduePair<Residue>: EntryTrait<LogPType = na::SVector<Float, DIM>>,
    Const<DIM>: Doubleable + Exponentiable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>,
{
    let (num_leaves, _residue_seq_length) = residue_sequences_2d.shape();
    let num_nodes = tree.len();

    let log_likelihood_total: Vec<Float>;
    let grad_rate_total: Vec<na::SMatrix<Float, DIM, DIM>>;
    let grad_log_prior_total: Vec<na::SVector<Float, DIM>>;
    (
        log_likelihood_total,
        (grad_rate_total, grad_log_prior_total),
    ) = (index_pairs, rate_matrices, log_p_priors)
        .into_par_iter()
        .map(|(column_id, rate_matrix, log_p_prior)| {
            let (left_id, right_id) = *column_id;
            let left_half = residue_sequences_2d.column(left_id);
            let right_half = residue_sequences_2d.column(right_id);
            let column_iter = std::iter::zip(left_half.iter(), right_half.iter()).map(
                |(left_residue, right_residue)| {
                    ResiduePair::<Residue>(*left_residue, *right_residue)
                },
            );

            let forward_data = forward_data_precompute(rate_matrix.as_view(), &distances);

            let log_p = forward_column(column_iter, distribution, &tree, &forward_data);
            let log_p_root = log_p[num_nodes - 1];

            let (log_likelihood_column, grad_log_p_likelihood): (Float, na::SVector<Float, DIM>) =
                process_likelihood(log_p_root.as_view(), log_p_prior.as_view());
            let grad_log_prior_column = grad_log_p_likelihood;
            let grad_log_p_root = grad_log_p_likelihood;

            let grad_rate_column = d_rate_column(
                grad_log_p_root.as_view(),
                &tree,
                &log_p,
                &distances,
                &forward_data,
                num_leaves,
            );

            (
                log_likelihood_column,
                (grad_rate_column, grad_log_prior_column),
            )
        })
        .unzip();
    (log_likelihood_total, grad_rate_total, grad_log_prior_total)
}

fn d_rate_column_param<const DIM: usize>(
    grad_log_p_root: na::SVectorView<Float, DIM>,
    tree: &[TreeNode],
    log_p: &[na::SVector<Float, DIM>],
    distances: &[Float],
    forward_data: &ForwardData<DIM>,
    param: &ParamData<DIM>,
    num_leaves: usize,
) -> na::SMatrix<Float, DIM, DIM> {
    /* Notice that child_input values are always added, so the log_p input for children is always the same.
    We will therefore store their common grad_log_p in the parent node's BackwardData. */
    /* TODO: it is possible to free grad_log_p's for the previous tree level. */
    let num_nodes = tree.len();
    let mut backward_data = Vec::<BackwardData<DIM>>::with_capacity(num_nodes - num_leaves);
    let mut grad_rate_column = na::SMatrix::<Float, DIM, DIM>::from_element(0.0 as Float);
    /* root.backward */
    backward_data.push(BackwardData {
        grad_log_p: grad_log_p_root.clone_owned(),
    });
    /* node.backward for non-terminal nodes */
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = &tree[id].parent;
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

pub fn train_parallel_param<const DIM: usize, Residue, D>(
    index_pairs: &[(usize, usize)],
    residue_sequences_2d: na::DMatrixView<Residue>,
    deltas: &[na::SMatrix<Float, DIM, DIM>],
    sqrt_pi: &[na::SVector<Float, DIM>],
    tree: &[TreeNode],
    distances: &[Float],
    distribution: &D,
) -> (
    Vec<Float>,
    Vec<na::SMatrix<Float, DIM, DIM>>,
    Vec<na::SVector<Float, DIM>>,
    Vec<na::SMatrix<Float, DIM, DIM>>,
)
where
    Residue: ResidueTrait,
    ResiduePair<Residue>: EntryTrait,
    D: Distribution<ResiduePair<Residue>, Float, DIM>,
{
    let (num_leaves, _residue_seq_length) = residue_sequences_2d.shape();
    let num_nodes = tree.len();

    let log_likelihood_total: Vec<Float>;
    let grad_delta_total: Vec<na::SMatrix<Float, DIM, DIM>>;
    let grad_sqrt_pi_total: Vec<na::SVector<Float, DIM>>;
    let grad_rate_total: Vec<na::SMatrix<Float, DIM, DIM>>;
    (
        log_likelihood_total,
        (grad_delta_total, (grad_sqrt_pi_total, grad_rate_total)),
    ) = (index_pairs, deltas, sqrt_pi)
        .into_par_iter()
        .map(|(column_id, delta, sqrt_pi)| {
            let (left_id, right_id) = *column_id;
            let left_half = residue_sequences_2d.column(left_id);
            let right_half = residue_sequences_2d.column(right_id);
            let column_iter = std::iter::zip(left_half.iter(), right_half.iter()).map(
                |(left_residue, right_residue)| {
                    ResiduePair::<Residue>(*left_residue, *right_residue)
                },
            );

            let param = compute_param_data(delta.as_view(), sqrt_pi.as_view());

            let forward_data = forward_data_precompute_param(&param, &distances);

            let log_p = forward_column(column_iter, distribution, &tree, &forward_data);
            let log_p_root = log_p[num_nodes - 1];

            let log_p_prior = sqrt_pi.map(Float::ln) * (2.0 as Float);
            let (log_likelihood_column, grad_log_p_likelihood): (Float, na::SVector<Float, DIM>) =
                process_likelihood(log_p_root.as_view(), log_p_prior.as_view());
            let grad_log_prior_column = grad_log_p_likelihood;
            let grad_log_p_root = grad_log_p_likelihood;

            let grad_rate_column = d_rate_column_param(
                grad_log_p_root.as_view(),
                &tree,
                &log_p,
                &distances,
                &forward_data,
                &param,
                num_leaves,
            );

            let (grad_delta_column, mut grad_sqrt_pi_column) =
                d_param(grad_rate_column.as_view(), &param);

            let mut grad_sqrt_pi_likelihood = param.sqrt_pi_recip * (2.0 as Float);
            grad_sqrt_pi_likelihood
                .component_mul_assign(&na::SVector::<Float, DIM>::from(grad_log_prior_column));
            grad_sqrt_pi_column += grad_sqrt_pi_likelihood;

            (
                log_likelihood_column,
                (grad_delta_column, (grad_sqrt_pi_column, grad_rate_column)),
            )
        })
        .unzip();
    (
        log_likelihood_total,
        grad_delta_total,
        grad_sqrt_pi_total,
        grad_rate_total,
    )
}
