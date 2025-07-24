use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::backward::*;
use crate::data_types::*;
use crate::forward::*;
use crate::tree::*;

use nalgebra as na;

/// leaf_log_p should be already big enough to contain log_p for all the nodes, not just the leaves.
/// We assume it is initialized with the leaf log probabilities.
fn forward_column<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &mut [na::SVector<F, DIM>],
    tree: &[TreeNode],
    forward_data: &ForwardData<F, DIM>,
    num_leaves: usize,
) {
    let num_nodes = tree.len();

    for i in num_leaves..(num_nodes - 1) {
        leaf_log_p[i] = forward_node(i, tree, &leaf_log_p, forward_data);
    }
    leaf_log_p[num_nodes - 1] = forward_root(num_nodes - 1, tree, &leaf_log_p, forward_data);
}

fn final_likelihood<F: FloatTrait, const DIM: usize>(
    log_p_root: na::SVectorView<F, DIM>,
    log_p_prior: na::SVectorView<F, DIM>,
) -> (F, na::SVector<F, DIM>) {
    let lse_arg = log_p_root + log_p_prior;
    let log_likelihood_column = F::logsumexp(lse_arg.iter());
    let grad_log_p_outgoing = softmax(&lse_arg);
    (log_likelihood_column, grad_log_p_outgoing)
}

fn d_rate_matrix<F: FloatTrait, const DIM: usize>(
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
    let mut grad_rate = na::SMatrix::<F, DIM, DIM>::zeros();
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = tree[id].parent;
        let parent_backward_id = num_nodes - parent_id as usize - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let distance_current = distances[id];
        let fwd_data_current = &forward_data.log_transition[id];
        let grad_log_p = d_child_input_param(
            grad_log_p_input,
            distance_current,
            param,
            log_p_input,
            fwd_data_current,
            true,
            &mut grad_rate,
        );
        grad_rate_column += grad_rate;
        backward_data.push(BackwardData {
            grad_log_p: grad_log_p.unwrap(),
        });
    }
    /* For leaves, we only compute grad_rate */
    for id in (0..num_leaves).rev() {
        let parent_id = tree[id].parent;
        let parent_backward_id = num_nodes - parent_id as usize - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let distance_current = distances[id];
        let fwd_data_current = &forward_data.log_transition[id];
        d_child_input_param(
            grad_log_p_input,
            distance_current,
            param,
            log_p_input,
            fwd_data_current,
            false,
            &mut grad_rate,
        );
        grad_rate_column += grad_rate;
    }
    param.V_pi_inv.tr_mul(&grad_rate_column) * param.V_pi.transpose()
}

pub struct SingleSideResult<F, const DIM: usize> {
    log_likelihood: F,
    grad_s: na::SMatrix<F, DIM, DIM>,
    grad_sqrt_pi: na::SVector<F, DIM>,
}

pub fn calculate_column<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &mut [na::SVector<F, DIM>],
    S: na::SMatrixView<F, DIM, DIM>,
    sqrt_pi: na::SVectorView<F, DIM>,
    tree: &[TreeNode],
    distances: &[F],
    num_leaves: usize,
) -> SingleSideResult<F, DIM> {
    // If the diagonalization fails or eigenvalues are to big, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S, sqrt_pi) {
        Some(param) => param,
        None => {
            return SingleSideResult::<F, DIM> {
                log_likelihood: <F as num_traits::Float>::neg_infinity(),
                grad_s: na::SMatrix::<F, DIM, DIM>::zeros(),
                grad_sqrt_pi: na::SVector::<F, DIM>::zeros(),
            }
        }
    };

    let forward_data = forward_data_precompute_param(&param, distances);

    forward_column(leaf_log_p, tree, &forward_data, num_leaves);
    let log_p = leaf_log_p;
    let log_p_root = log_p.last().unwrap();

    let log_p_prior = sqrt_pi.map(num_traits::Float::ln) * <F as FloatTrait>::from_f64(2.0);
    let (log_likelihood, grad_log_p_likelihood): (F, na::SVector<F, DIM>) =
        final_likelihood(log_p_root.as_view(), log_p_prior.as_view());
    let grad_log_prior = grad_log_p_likelihood;
    let grad_log_p_root = grad_log_p_likelihood;

    let grad_rate = d_rate_matrix(
        grad_log_p_root.as_view(),
        tree,
        &log_p,
        distances,
        &forward_data,
        &param,
        num_leaves,
    );

    let (grad_s, mut grad_sqrt_pi) = d_param(grad_rate.as_view(), &param);

    let mut grad_sqrt_pi_likelihood: na::SMatrix<F, DIM, 1> =
        param.sqrt_pi_recip * <F as FloatTrait>::from_f64(2.0);
    grad_sqrt_pi_likelihood.component_mul_assign(&grad_log_prior);
    grad_sqrt_pi += grad_sqrt_pi_likelihood;
    SingleSideResult::<F, DIM> {
        log_likelihood,
        grad_s,
        grad_sqrt_pi,
    }
}

#[derive(Debug)]
pub struct FelsensteinResult<F, const DIM: usize> {
    pub log_likelihood: Vec<F>,
    pub grad_s: Vec<na::SMatrix<F, DIM, DIM>>,
    pub grad_sqrt_pi: Vec<na::SVector<F, DIM>>,
}

pub fn calculate_column_parallel<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &mut [Vec<na::SVector<F, DIM>>],
    S: &[na::SMatrix<F, DIM, DIM>],
    sqrt_pi: &[na::SVector<F, DIM>],
    tree: &[TreeNode],
    distances: &[F],
    num_leaves: usize,
) -> FelsensteinResult<F, DIM> {
    let col_results = (leaf_log_p, S, sqrt_pi)
        .into_par_iter()
        .map(|(mut leaf_log_p, S, sqrt_pi)| {
            calculate_column(
                &mut leaf_log_p,
                S.as_view(),
                sqrt_pi.as_view(),
                tree,
                distances,
                num_leaves,
            )
        })
        .collect::<Vec<_>>();

    let mut log_likelihood_total = vec![];
    let mut grad_delta_total = vec![];
    let mut grad_sqrt_pi_total = vec![];

    for col_result in col_results {
        log_likelihood_total.push(col_result.log_likelihood);
        grad_delta_total.push(col_result.grad_s);
        grad_sqrt_pi_total.push(col_result.grad_sqrt_pi);
    }

    FelsensteinResult {
        log_likelihood: log_likelihood_total,
        grad_s: grad_delta_total,
        grad_sqrt_pi: grad_sqrt_pi_total,
    }
}

/// Same as `calculate_column_parallel`, but a single S and sqrt_pi are passed and used for all sides. It still produces a FelsensteinResult with Vec of length 1.
/// This is significantly faster than calculate_column_parallel.
pub fn calculate_column_parallel_single_S<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &mut [Vec<na::SVector<F, DIM>>],
    S: &na::SMatrix<F, DIM, DIM>,
    sqrt_pi: &na::SVector<F, DIM>,
    tree: &[TreeNode],
    distances: &[F],
    d_trans_matrix: &mut [Vec<na::SMatrix<F, DIM, DIM>>],
    num_leaves: usize,
) -> FelsensteinResult<F, DIM> {
    let L = leaf_log_p.len();

    // If lapack fails to diaginalize or the eigenvalues are too extreme, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S.as_view(), sqrt_pi.as_view()) {
        Some(param) => param,
        None => {
            return FelsensteinResult::<F, DIM> {
                log_likelihood: vec![<F as num_traits::Float>::neg_infinity(); L],
                grad_s: vec![na::SMatrix::<F, DIM, DIM>::zeros()],
                grad_sqrt_pi: vec![na::SVector::<F, DIM>::zeros()],
            }
        }
    };

    let forward_data = forward_data_precompute_param(&param, distances);

    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let result = leaf_log_p
        .into_par_iter()
        .zip(d_trans_matrix.par_iter_mut())
        .map(|(leaf_log_p, d_trans)| {
            cacluate_column_single_S(leaf_log_p, &param, &forward_data, tree, d_trans, num_leaves)
        })
        .collect::<Vec<_>>();

    let log_likelihood = result.iter().map(|r| r.0).collect::<Vec<_>>();

    let sum_d_log_prior = result.iter().map(|r| r.1).sum::<na::SVector<F, DIM>>();

    let d_rate_matrix = forward_data
        .log_transition
        .into_par_iter()
        .enumerate()
        .map(|(idx, forward)| {
            d_rate_matrix_per_edge(d_trans_matrix, idx, distances[idx], &param, &forward)
        })
        .sum::<na::SMatrix<F, DIM, DIM>>();

    let d_rate_matrix = param.V_pi_inv.tr_mul(&d_rate_matrix) * param.V_pi.transpose();

    let (grad_s, mut grad_sqrt_pi) = d_param(d_rate_matrix.as_view(), &param);

    let mut grad_sqrt_pi_likelihood: na::SMatrix<F, DIM, 1> =
        param.sqrt_pi_recip * <F as FloatTrait>::from_f64(2.0);
    grad_sqrt_pi_likelihood.component_mul_assign(&sum_d_log_prior);
    grad_sqrt_pi += grad_sqrt_pi_likelihood;

    FelsensteinResult::<F, DIM> {
        log_likelihood,
        grad_s: vec![grad_s],
        grad_sqrt_pi: vec![grad_sqrt_pi],
    }
}

fn d_rate_matrix_per_edge<F: FloatTrait, const DIM: usize>(
    d_trans_matrix: &[Vec<na::SMatrix<F, DIM, DIM>>],
    edge: usize,
    distance: F,
    param: &ParamPrecomp<F, DIM>,
    forward: &LogTransitionForwardData<F, DIM>,
) -> na::SMatrix<F, DIM, DIM> {
    let mut sum_d_log_trans = d_trans_matrix.iter().map(|d_trans| d_trans[edge]).sum();

    d_ln_vjp(&mut sum_d_log_trans, &forward.matrix_exp);
    d_expm_vjp(&mut sum_d_log_trans, distance, param, &forward.exp_t_lambda);
    sum_d_log_trans
}

fn cacluate_column_single_S<F: FloatTrait, const DIM: usize>(
    leaf_log_p: &mut [na::SVector<F, DIM>],
    param: &ParamPrecomp<F, DIM>,
    forward_data: &ForwardData<F, DIM>,
    tree: &[TreeNode],
    d_trans_matrix: &mut [na::SMatrix<F, DIM, DIM>],
    num_leaves: usize,
) -> (F, na::SVector<F, DIM>) {
    forward_column(leaf_log_p, tree, forward_data, num_leaves);
    let log_p = leaf_log_p;
    let log_p_root = log_p.last().unwrap();

    let log_p_prior = param.sqrt_pi.map(num_traits::Float::ln) * <F as FloatTrait>::from_f64(2.0);

    let (log_likelihood, grad_log_p_likelihood) =
        final_likelihood(log_p_root.as_view(), log_p_prior.as_view());
    let d_log_prior = grad_log_p_likelihood;
    let d_log_p_root = grad_log_p_likelihood;

    d_trans_matrix_fn(
        d_log_p_root.as_view(),
        tree,
        &log_p,
        forward_data,
        num_leaves,
        d_trans_matrix,
    );

    (log_likelihood, d_log_prior)
}

/// Write the gradient of the transition matrix into d_trans
/// We sum over all the columns later
fn d_trans_matrix_fn<F: FloatTrait, const DIM: usize>(
    grad_log_p_root: na::SVectorView<F, DIM>,
    tree: &[TreeNode],
    log_p: &[na::SVector<F, DIM>],
    forward_data: &ForwardData<F, DIM>,
    num_leaves: usize,
    d_trans: &mut [na::SMatrix<F, DIM, DIM>],
) {
    /* Notice that child_input values are always added, so the log_p input for children is always the same.
    We will therefore store their common grad_log_p in the parent node's BackwardData. */
    /* TODO: it is possible to free grad_log_p's for the previous tree level. */
    let num_nodes = tree.len();
    let mut backward_data = Vec::<BackwardData<F, DIM>>::with_capacity(num_nodes - num_leaves);
    /* root.backward */
    backward_data.push(BackwardData {
        grad_log_p: grad_log_p_root.clone_owned(),
    });
    /* node.backward for non-terminal nodes */
    for id in (num_leaves..num_nodes - 1).rev() {
        let parent_id = tree[id].parent;
        let parent_backward_id = num_nodes - parent_id as usize - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let fwd_data_current = &forward_data.log_transition[id];
        let grad_log_p = d_log_transition_child_input_vjp(
            grad_log_p_input,
            log_p_input,
            fwd_data_current,
            true,
            &mut d_trans[id],
        );
        backward_data.push(BackwardData {
            grad_log_p: grad_log_p.unwrap(),
        });
    }
    /* For leaves, we only compute grad_rate */
    for id in (0..num_leaves).rev() {
        let parent_id = tree[id].parent;
        let parent_backward_id = num_nodes - parent_id as usize - 1;
        let grad_log_p_input = backward_data[parent_backward_id].grad_log_p.as_view();
        let log_p_input = log_p[id].as_view();
        let fwd_data_current = &forward_data.log_transition[id];
        d_log_transition_child_input_vjp(
            grad_log_p_input,
            log_p_input,
            fwd_data_current,
            false,
            &mut d_trans[id],
        );
    }
}
