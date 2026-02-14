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

/// final likelihood given the root partial_likelihood and the prior distribution
/// also returns the gradient of the likelihood with respect to the root partial likelihood and sqrt_pi
fn final_likelihood<const DIM: usize>(
    lin_pl_root: na::SVectorView<f64, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
) -> (f64, na::SVector<f64, DIM>, na::SVector<f64, DIM>) {
    let pi = sqrt_pi.component_mul(&sqrt_pi);
    let likelihood = pi.dot(&lin_pl_root);
    
    (likelihood.ln(), pi / likelihood, (lin_pl_root.component_mul(&sqrt_pi) * 2.0) / likelihood)
}

pub struct SingleSideResult<F, const DIM: usize> {
    pub log_likelihood: F,
    pub grad_s: na::SMatrix<F, DIM, DIM>,
    pub grad_sqrt_pi: na::SVector<F, DIM>,
}

pub fn calculate_column<const DIM: usize>(
    log_p: &mut [na::SVector<f64, DIM>],
    S: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    tree: Tree,
    only_likelihood: bool,
) -> SingleSideResult<f64, DIM> {
    // If the diagonalization fails or eigenvalues are to big, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S, sqrt_pi) {
        Some(param) => param,
        None => {
            return SingleSideResult::<f64, DIM> {
                log_likelihood: f64::NEG_INFINITY,
                grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
                grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
            }
        }
    };

    let forward_data = forward_data_precompute_param(&param, tree.distances);
    forward_column(log_p, tree.parents, &forward_data);
    let lin_pl_root = log_p.last().unwrap();

    let (log_likelihood, d_lin_pl_root, d_sqrt_pi) =
        final_likelihood(lin_pl_root.as_view(), sqrt_pi.as_view());


    if only_likelihood {
        return SingleSideResult::<f64, DIM> {
            log_likelihood,
            grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
            grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
        };
    }

    let d_Q = d_Q(&d_lin_pl_root, tree, log_p, &param, &forward_data.model_edge_data);

    let (grad_s, mut grad_sqrt_pi) = d_param(d_Q.as_view(), &param);

    grad_sqrt_pi += d_sqrt_pi;
    SingleSideResult::<f64, DIM> {
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
    const DIM: usize,
    A: AsMut<[na::SVector<f64, DIM>]> + Send,
>(
    leaf_log_p: &mut [A],
    S: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    tree: Tree,
    only_likelihood: bool,
) -> FelsensteinResult<f64, DIM> {
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
fn d_Q<const DIM: usize>(
    grad_p_root: &na::SVector<f64, DIM>,
    tree: Tree,
    lin_pl: &[na::SVector<f64, DIM>],
    param: &ParamPrecomp<DIM>,
    forward: &[ModelEdgeData<f64, DIM>],
) -> na::SMatrix<f64, DIM, DIM> {
    let top_bifurcations = get_topological_bifurcations(&tree);
    let mut cotangents = vec![na::SVector::<f64, DIM>::zeros(); tree.parents.len()];
    cotangents.last_mut().unwrap().copy_from(&grad_p_root);

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    for bi in top_bifurcations.into_iter().rev() {
        backward::d_log_transition_bifurcation_vjp(&mut cotangents, lin_pl, forward, param, &mut d_Q, &bi, &tree.distances);
    }

    param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose()
}
