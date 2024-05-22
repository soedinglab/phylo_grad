use logsumexp::LogSumExp;
use na::{Const, DefaultAllocator};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::backward::*;
use crate::data_types::*;
use crate::forward::*;
use crate::tree::*;

fn forward_column<const DIM: usize, Entry>(
    column_iter: impl Iterator<Item = Entry>,
    tree: &[TreeNode],
    forward_data: &ForwardData<DIM>,
) -> Vec<[Float; DIM]>
where
    Entry: EntryTrait<LogPType = [Float; DIM]>,
{
    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    let num_nodes = tree.len();
    let mut log_p = Vec::<[Float; DIM]>::with_capacity(num_nodes);
    log_p.extend(column_iter.map(|x| Entry::to_log_p(&x)));
    let num_leaves = log_p.len();

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
    log_p_root: &[Float; DIM],
    log_p_prior: &[Float; DIM],
) -> (Float, [Float; DIM]) {
    let mut forward_data_likelihood_lse_arg = [0.0 as Float; DIM];
    for i in 0..DIM {
        forward_data_likelihood_lse_arg[i] = log_p_root[i] + log_p_prior[i];
    }
    let log_likelihood_column = forward_data_likelihood_lse_arg.iter().ln_sum_exp();

    softmax_inplace(&mut forward_data_likelihood_lse_arg);
    let grad_log_p_outgoing = forward_data_likelihood_lse_arg;
    (log_likelihood_column, grad_log_p_outgoing)
}

fn d_rate_column<const DIM: usize>(
    grad_log_p_root: &[Float; DIM],
    tree: &[TreeNode],
    log_p: &[[Float; DIM]],
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
        grad_log_p: *grad_log_p_root,
    });
    /* node.backward for non-terminal nodes */
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = &tree[id].parent;
        let parent_backward_id = num_nodes - parent_id - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p;
        let log_p_input = &log_p[id];
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
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p;
        let log_p_input = &log_p[id];
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

/* TODO should we let rayon know that index_paris is a vector and not just any slice? */
/* TODO Id<DIM> */
pub fn train_parallel<const DIM: usize, Residue>(
    index_pairs: &Vec<(usize, usize)>,
    residue_sequences_2d: na::DMatrixView<Residue>,
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    log_p_prior: &[Float; DIM],
    tree: &[TreeNode],
    distances: &[Float],
) -> (
    Vec<Float>,
    Vec<na::SMatrix<Float, DIM, DIM>>,
    Vec<[Float; DIM]>,
)
where
    Residue: ResidueTrait,
    ResiduePair<Residue>: EntryTrait<LogPType = [Float; DIM]>,
    Const<DIM>: Doubleable + Exponentiable, //+ Decrementable,
    TwoTimesConst<DIM>: Exponentiable,
    DefaultAllocator: ViableAllocator<Float, DIM>, //+ DecrementableAllocator<Float, DIM>,
{
    let (num_leaves, _residue_seq_length) = residue_sequences_2d.shape();
    let num_nodes = tree.len();

    let log_likelihood_total: Vec<Float>;
    let grad_rate_total: Vec<na::SMatrix<Float, DIM, DIM>>;
    let grad_log_prior_total: Vec<[Float; DIM]>;
    (
        log_likelihood_total,
        (grad_rate_total, grad_log_prior_total),
    ) = index_pairs
        .par_iter()
        .map(|column_id| {
            let (left_id, right_id) = *column_id;
            let left_half = residue_sequences_2d.column(left_id);
            let right_half = residue_sequences_2d.column(right_id);
            let column_iter = std::iter::zip(left_half.iter(), right_half.iter()).map(
                |(left_residue, right_residue)| {
                    ResiduePair::<Residue>(*left_residue, *right_residue)
                },
            );

            /* Right now, this is the same for all columns, but as every column will have its own
            rate matrix, in general we'll have to precompute log_transition for each column */
            let forward_data = forward_data_precompute(rate_matrix.as_view(), &distances);

            let log_p = forward_column(column_iter, &tree, &forward_data);
            let log_p_root = log_p[num_nodes - 1];

            let (log_likelihood_column, grad_log_p_likelihood): (Float, [Float; DIM]) =
                process_likelihood(&log_p_root, log_p_prior);
            let grad_log_prior_column = grad_log_p_likelihood;
            let grad_log_p_root = grad_log_p_likelihood;

            let mut grad_rate_column = d_rate_column(
                &grad_log_p_root,
                &tree,
                &log_p,
                &distances,
                &forward_data,
                num_leaves,
            );

            /* Gradient is differential transposed */
            grad_rate_column.transpose_mut();

            (
                log_likelihood_column,
                (grad_rate_column, grad_log_prior_column),
            )
        })
        .unzip();
    (log_likelihood_total, grad_rate_total, grad_log_prior_total)
}
