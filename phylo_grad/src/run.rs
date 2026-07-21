use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::backward::{self, *};
use crate::forward::*;
use crate::tree::*;

use nalgebra as na;

/// log_p should have the leaf log_p initialized and all the other nodes set to zero
fn forward_column<const DIM: usize>(
    lin_partial_likelihoods: &mut [na::SVector<f64, DIM>],
    parents: &[i32],
    offsets: &mut [u32],
    forward_data: &[ModelEdgeData<DIM>],
    param: &ParamPrecomp<DIM>,
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
            &param,
            offsets,
        );
    }
}

/// final likelihood given the root partial_likelihood and the prior distribution
/// also returns the gradient of the likelihood with respect to the root partial likelihood and sqrt_pi
/// The real gradients at root_partial_likelihood are 2 ** height bigger.
fn final_likelihood<const DIM: usize>(
    lin_pl_root: na::SVectorView<f64, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    root_height: u32,
) -> (f64, na::SVector<f64, DIM>, na::SVector<f64, DIM>) {
    let pi = sqrt_pi.component_mul(&sqrt_pi);
    let likelihood = pi.dot(&lin_pl_root);

    (
        likelihood.ln() - root_height as f64 * f64::ln(2.0),
        pi / likelihood,
        (lin_pl_root.component_mul(&sqrt_pi) * 2.0) / likelihood,
    )
}

#[derive(Debug, Clone)]
pub struct SingleSideResult<F, const DIM: usize> {
    pub log_likelihood: F,
    pub grad_s: na::SMatrix<F, DIM, DIM>,
    pub grad_sqrt_pi: na::SVector<F, DIM>,
}

pub fn calculate_column<const DIM: usize>(
    pl: &mut [na::SVector<f64, DIM>],
    S: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    tree: Tree,
    only_likelihood: bool,
    grad_edge_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    // If the diagonalization fails or eigenvalues are to big, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S, sqrt_pi) {
        Some(param) => param,
        None => {
            return SingleSideResult::<f64, DIM> {
                log_likelihood: f64::NEG_INFINITY,
                grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
                grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
            };
        }
    };

    let edge_data = forward_data_precompute_param(&param, tree.distances);
    let mut offsets = vec![0; tree.parents.len()];
    forward_column(pl, tree.parents, &mut offsets, &edge_data, &param);
    let lin_pl_root = pl.last().unwrap();

    let root_offset: u32 = offsets.iter().sum();

    let (log_likelihood, d_lin_pl_root, d_sqrt_pi) =
        final_likelihood(lin_pl_root.as_view(), sqrt_pi.as_view(), root_offset);

    if only_likelihood {
        return SingleSideResult::<f64, DIM> {
            log_likelihood,
            grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
            grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
        };
    }

    let d_Q = d_Q(
        &d_lin_pl_root,
        tree,
        pl,
        &param,
        &edge_data,
        &offsets,
        grad_edge_lengths,
    );

    let (grad_s, mut grad_sqrt_pi) = d_param(d_Q.as_view(), &param);

    grad_sqrt_pi += d_sqrt_pi;
    SingleSideResult::<f64, DIM> {
        log_likelihood,
        grad_s,
        grad_sqrt_pi,
    }
}

#[derive(Debug, Clone)]
pub struct FelsensteinResult<const DIM: usize> {
    pub log_likelihood: Vec<f64>,
    pub grad_s: Vec<na::SMatrix<f64, DIM, DIM>>,
    pub grad_sqrt_pi: Vec<na::SVector<f64, DIM>>,
}

#[derive(Debug, Clone)]
pub struct FelsensteinResultWithTree<const DIM: usize> {
    pub log_likelihood: Vec<f64>,
    pub grad_s: Vec<na::SMatrix<f64, DIM, DIM>>,
    pub grad_sqrt_pi: Vec<na::SVector<f64, DIM>>,
    pub grad_tree: Vec<Vec<f64>>,
}

pub fn calculate_column_parallel<const DIM: usize, A: AsMut<[na::SVector<f64, DIM>]> + Send>(
    leaf_pl: &mut [A],
    S: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    tree: Tree,
    only_likelihood: bool,
    tree_grad: &mut Option<Vec<Vec<f64>>>,
) -> FelsensteinResult<DIM> {
    if tree_grad.is_some() {
        let tree_grads = tree_grad.as_mut().unwrap();

        let col_results = (leaf_pl, S, sqrt_pi, tree_grads)
            .into_par_iter()
            .map(|(leaf_log_p, S, sqrt_pi, mut tree_grads)| {
                let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
                calculate_column(
                    leaf_log_p.as_mut(),
                    S.as_view(),
                    sqrt_pi.as_view(),
                    tree.clone(),
                    only_likelihood,
                    Some(&mut tree_grads),
                ) // The clone is shallow, Tree is cheap to clone
            })
            .collect::<Vec<_>>();

        let mut log_likelihood_total = vec![];
        let mut grad_s_total = vec![];
        let mut grad_sqrt_pi_total = vec![];

        for col_result in col_results.iter() {
            log_likelihood_total.push(col_result.log_likelihood);
            grad_s_total.push(col_result.grad_s);
            grad_sqrt_pi_total.push(col_result.grad_sqrt_pi);
        }

        FelsensteinResult {
            log_likelihood: log_likelihood_total,
            grad_s: grad_s_total,
            grad_sqrt_pi: grad_sqrt_pi_total,
        }
    } else {
        let col_results = (leaf_pl, S, sqrt_pi)
            .into_par_iter()
            .map(|(leaf_log_p, S, sqrt_pi)| {
                let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
                calculate_column(
                    leaf_log_p.as_mut(),
                    S.as_view(),
                    sqrt_pi.as_view(),
                    tree.clone(),
                    only_likelihood,
                    None,
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
}

/// For one column
fn d_Q<const DIM: usize>(
    grad_p_root: &na::SVector<f64, DIM>,
    tree: Tree,
    lin_pl: &[na::SVector<f64, DIM>],
    param: &ParamPrecomp<DIM>,
    forward: &[ModelEdgeData<DIM>],
    offsets: &[u32],
    mut d_edge_lengths: Option<&mut [f64]>,
) -> na::SMatrix<f64, DIM, DIM> {
    let top_bifurcations = get_topological_bifurcations(&tree);
    let mut cotangents = vec![na::SVector::<f64, DIM>::zeros(); tree.parents.len()];
    cotangents.last_mut().unwrap().copy_from(&grad_p_root);

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    for bi in top_bifurcations.iter().rev() {
        backward::d_log_transition_bifurcation_vjp(
            &mut cotangents,
            lin_pl,
            forward,
            param,
            &mut d_Q,
            bi,
            &tree.distances,
            offsets,
            d_edge_lengths.as_deref_mut(),
        );
    }

    param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose()
}

fn calculate_column_with_precompute<const DIM: usize>(
    pl: &mut [na::SVector<f64, DIM>],
    param: &ParamPrecomp<DIM>,
    forward_data: &[ModelEdgeData<DIM>],
    tree: Tree,
    only_likelihood: bool,
    grad_edge_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    let mut offsets = vec![0; tree.parents.len()];
    forward_column(pl, tree.parents, &mut offsets, forward_data, param);

    let lin_pl_root = pl.last().unwrap();

    let root_offset: u32 = offsets.iter().sum();

    let (log_likelihood, d_lin_pl_root, d_sqrt_pi) =
        final_likelihood(lin_pl_root.as_view(), param.sqrt_pi.as_view(), root_offset);

    if only_likelihood {
        return SingleSideResult::<f64, DIM> {
            log_likelihood,
            grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
            grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
        };
    }

    let d_Q = d_Q(
        &d_lin_pl_root,
        tree,
        pl,
        param,
        &forward_data,
        &offsets,
        grad_edge_lengths,
    );

    let (grad_s, mut grad_sqrt_pi) = d_param(d_Q.as_view(), param);
    grad_sqrt_pi += d_sqrt_pi;

    SingleSideResult::<f64, DIM> {
        log_likelihood,
        grad_s,
        grad_sqrt_pi,
    }
}

pub fn calculate_columns_single_S_parallel<const DIM: usize>(
    pl: &mut [Vec<na::SVector<f64, DIM>>],
    S: &na::SMatrix<f64, DIM, DIM>,
    sqrt_pi: &na::SVector<f64, DIM>,
    tree: Tree,
    only_likelihood: bool,
    d_edge_lengths: Option<&mut [Vec<f64>]>,
) -> FelsensteinResult<DIM> {
    let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));

    let param = match compute_param_data(S.as_view(), sqrt_pi.as_view()) {
        Some(param) => param,
        None => {
            return FelsensteinResult::<DIM> {
                log_likelihood: vec![f64::NEG_INFINITY; pl.len()],
                grad_s: vec![na::SMatrix::<f64, DIM, DIM>::zeros(); pl.len()],
                grad_sqrt_pi: vec![na::SVector::<f64, DIM>::zeros(); pl.len()],
            };
        }
    };

    let forward_data = forward_data_precompute_param(&param, tree.distances);

    if let Some(tree_grads) = d_edge_lengths {
        let col_results = (pl, tree_grads)
            .into_par_iter()
            .map(|(leaf_log_p, tree_grads)| {
                calculate_column_with_precompute(
                    leaf_log_p,
                    &param,
                    &forward_data,
                    tree.clone(),
                    only_likelihood,
                    Some(tree_grads),
                )
            })
            .collect::<Vec<_>>();

        let mut log_likelihood_total = vec![];
        let mut grad_s_total = vec![];
        let mut grad_sqrt_pi_total = vec![];

        for col_result in col_results {
            log_likelihood_total.push(col_result.log_likelihood);
            grad_s_total.push(col_result.grad_s);
            grad_sqrt_pi_total.push(col_result.grad_sqrt_pi);
        }

        FelsensteinResult {
            log_likelihood: log_likelihood_total,
            grad_s: grad_s_total,
            grad_sqrt_pi: grad_sqrt_pi_total,
        }
    } else {
        let col_results = pl
            .into_par_iter()
            .map(|leaf_log_p| {
                calculate_column_with_precompute(
                    leaf_log_p,
                    &param,
                    &forward_data,
                    tree.clone(),
                    only_likelihood,
                    None,
                )
            })
            .collect::<Vec<_>>();

        let mut log_likelihood_total = vec![];
        let mut grad_s_total = vec![];
        let mut grad_sqrt_pi_total = vec![];

        for col_result in col_results {
            log_likelihood_total.push(col_result.log_likelihood);
            grad_s_total.push(col_result.grad_s);
            grad_sqrt_pi_total.push(col_result.grad_sqrt_pi);
        }

        FelsensteinResult {
            log_likelihood: log_likelihood_total,
            grad_s: grad_s_total,
            grad_sqrt_pi: grad_sqrt_pi_total,
        }
    }
}
