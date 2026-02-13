use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::backward::{self, *};
use crate::data_types::*;
use crate::forward::*;
use crate::tree::*;

use nalgebra as na;

/// log_p should have the leaf log_p initialized and all the other nodes set to zero
fn forward_column<F: FloatTrait, const DIM: usize>(
    lin_partial_likelihoods: &mut [na::SVector<F, DIM>],
    parents: &[i32],
    forward_data: &ForwardData<F, DIM>,
) {
    for (child, &parent) in parents.iter().enumerate() {
        if parent == -1 {
            continue; // skip the root
        }
        forward_node(
            child as usize,
            parent as usize,
            lin_partial_likelihoods,
            forward_data,
        );
    }
}

/// final likelihood given the root partial_likelihood oand the prior distribution
fn final_likelihood<F: FloatTrait, const DIM: usize>(
    lin_pl_root: na::SVectorView<F, DIM>,
    log_p_prior: na::SVectorView<F, DIM>,
) -> (F, na::SVector<F, DIM>) {
    let lse_arg = lin_pl_root.map(|x| num_traits::Float::ln(x)) + log_p_prior;
    let log_likelihood_column = F::logsumexp(lse_arg.iter());
    let grad_log_p_outgoing = softmax(&lse_arg);
    (log_likelihood_column, grad_log_p_outgoing)
}

pub struct SingleSideResult<F, const DIM: usize> {
    pub log_likelihood: F,
    pub grad_s: na::SMatrix<F, DIM, DIM>,
    pub grad_sqrt_pi: na::SVector<F, DIM>,
}

pub fn calculate_column<F: FloatTrait, const DIM: usize>(
    log_p: &mut [na::SVector<F, DIM>],
    S: na::SMatrixView<F, DIM, DIM>,
    sqrt_pi: na::SVectorView<F, DIM>,
    tree: Tree<F>,
    only_likelihood: bool,
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

    let forward_data = forward_data_precompute_param(&param, tree.distances);
    forward_column(log_p, tree.parents, &forward_data);
    let lin_p_root = log_p.last().unwrap();

    let log_p_prior = sqrt_pi.map(num_traits::Float::ln) * <F as FloatTrait>::from_f64(2.0);
    let (log_likelihood, grad_log_p_likelihood) =
        final_likelihood(lin_p_root.as_view(), log_p_prior.as_view());


    if only_likelihood {
        return SingleSideResult::<F, DIM> {
            log_likelihood,
            grad_s: na::SMatrix::<F, DIM, DIM>::zeros(),
            grad_sqrt_pi: na::SVector::<F, DIM>::zeros(),
        };
    }
    let grad_p_likelihood = grad_log_p_likelihood.map(|x| num_traits::Float::recip(x));

    let grad_log_prior = grad_log_p_likelihood;

    let d_Q = d_Q(&grad_p_likelihood, tree, log_p, &param, &forward_data.model_edge_data);

    let (grad_s, mut grad_sqrt_pi) = d_param(d_Q.as_view(), &param);

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

pub fn calculate_column_parallel<
    F: FloatTrait,
    const DIM: usize,
    A: AsMut<[na::SVector<F, DIM>]> + Send,
>(
    leaf_log_p: &mut [A],
    S: &[na::SMatrix<F, DIM, DIM>],
    sqrt_pi: &[na::SVector<F, DIM>],
    tree: Tree<F>,
    only_likelihood: bool,
) -> FelsensteinResult<F, DIM> {
    let col_results = (leaf_log_p, S, sqrt_pi)
        .into_par_iter()
        .map(|(leaf_log_p, S, sqrt_pi)| {
            calculate_column(
                leaf_log_p.as_mut(),
                S.as_view(),
                sqrt_pi.as_view(),
                tree.clone(),
                only_likelihood,
            ) // The clone is shallow, Tree is cheap to clone
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

/// For one column
fn d_Q<F: FloatTrait, const DIM: usize>(
    grad_p_root: &na::SVector<F, DIM>,
    tree: Tree<F>,
    lin_pl: &[na::SVector<F, DIM>],
    param: &ParamPrecomp<F, DIM>,
    forward: &[ModelEdgeData<F, DIM>],
) -> na::SMatrix<F, DIM, DIM> {
    let top_bifurcations = get_topological_bifurcations(&tree);
    let mut cotangents = vec![na::SVector::<F, DIM>::zeros(); tree.parents.len()];
    cotangents.last_mut().unwrap().copy_from(&grad_p_root);

    let mut d_Q = na::SMatrix::<F, DIM, DIM>::zeros();

    for bi in top_bifurcations {
        backward::d_log_transition_bifurcation_vjp(&mut cotangents, lin_pl, forward, param, &mut d_Q, &bi, &tree.distances);
    }

    param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose()
}
