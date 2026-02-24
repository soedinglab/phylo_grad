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
    forward_data: &ForwardData<DIM>,
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

    let forward_data = forward_data_precompute_param(&param, tree.distances);
    let mut offsets = vec![0; tree.parents.len()];
    forward_column(pl, tree.parents, &mut offsets, &forward_data);
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
        &forward_data.model_edge_data,
        &offsets,
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

pub fn calculate_column_parallel<const DIM: usize, A: AsMut<[na::SVector<f64, DIM>]> + Send>(
    leaf_log_p: &mut [A],
    S: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    tree: Tree,
    only_likelihood: bool,
) -> FelsensteinResult<DIM> {
    let col_results = (leaf_log_p, S, sqrt_pi)
        .into_par_iter()
        .map(|(leaf_log_p, S, sqrt_pi)| {
            let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
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
    forward: &[ModelEdgeData<DIM>],
    offsets: &[u32],
) -> na::SMatrix<f64, DIM, DIM> {
    let top_bifurcations = get_topological_bifurcations(&tree);
    let mut cotangents = vec![na::SVector::<f64, DIM>::zeros(); tree.parents.len()];
    cotangents.last_mut().unwrap().copy_from(&grad_p_root);

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    for bi in top_bifurcations.into_iter().rev() {
        backward::d_log_transition_bifurcation_vjp(
            &mut cotangents,
            lin_pl,
            forward,
            param,
            &mut d_Q,
            &bi,
            &tree.distances,
            offsets,
            None,
        );
    }

    param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose()
}

struct TLS<const DIM: usize> {
    offsets: Vec<u32>,
    d_trans: Vec<na::SMatrix<f64, DIM, DIM>>,
    cotangents: Vec<na::SVector<f64, DIM>>,
}

impl<const DIM: usize> TLS<DIM> {
    fn new(num_nodes: usize) -> Self {
        TLS {
            offsets: vec![0; num_nodes],
            d_trans: vec![na::SMatrix::<f64, DIM, DIM>::zeros(); num_nodes],
            cotangents: vec![na::SVector::<f64, DIM>::zeros(); num_nodes],
        }
    }
}

fn cacluate_column_single_S<const DIM: usize>(
    pl: &mut [na::SVector<f64, DIM>],
    param: &ParamPrecomp<DIM>,
    forward_data: &ForwardData<DIM>,
    tree: Tree,
    only_likelihood: bool,
    tls: &mut TLS<DIM>,
    bifurcations: &[Bifurcation],
) -> (f64, na::SVector<f64, DIM>) {
    forward_column(pl, tree.parents, &mut tls.offsets, forward_data);

    let lin_pl_root = pl.last().unwrap();

    let root_offset: u32 = tls.offsets.iter().sum();

    let (log_likelihood, d_lin_pl_root, d_sqrt_pi) =
        final_likelihood(lin_pl_root.as_view(), param.sqrt_pi.as_view(), root_offset);

    if only_likelihood {
        return (log_likelihood, na::SVector::<f64, DIM>::zeros());
    }

    tls.cotangents.last_mut().unwrap().copy_from(&d_lin_pl_root);

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    for bi in bifurcations.iter() {
        backward::d_log_transition_bifurcation_vjp(
            &mut tls.cotangents,
            pl,
            &forward_data.model_edge_data,
            param,
            &mut d_Q,
            bi,
            &tree.distances,
            &tls.offsets,
            Some(&mut tls.d_trans),
        );
    }

    (log_likelihood, d_sqrt_pi)
}

pub fn calculate_column_block_single_S<const DIM: usize>(
    pl: &mut [Vec<na::SVector<f64, DIM>>],
    param: &ParamPrecomp<DIM>,
    forward_data: &ForwardData<DIM>,
    tree: Tree,
    only_likelihood: bool,
    bifurcations: &[Bifurcation],
) -> FelsensteinResult<DIM> {
    let mut tls = TLS::<DIM>::new(tree.parents.len());

    let mut log_likelihoods = Vec::with_capacity(pl.len());

    let mut d_sqrt_pi_sum = na::SVector::<f64, DIM>::zeros();

    for pl in pl.iter_mut() {
        tls.offsets.iter_mut().for_each(|o| *o = 0);
        tls.cotangents
            .iter_mut()
            .for_each(|c| *c = na::SVector::<f64, DIM>::zeros());
        let (ll, d_sqrt_pi) = cacluate_column_single_S(
            pl,
            param,
            forward_data,
            tree.clone(),
            only_likelihood,
            &mut tls,
            bifurcations,
        );
        d_sqrt_pi_sum += d_sqrt_pi;
        log_likelihoods.push(ll);
    }

    if only_likelihood {
        return FelsensteinResult {
            log_likelihood: log_likelihoods,
            grad_s: vec![na::SMatrix::<f64, DIM, DIM>::zeros(); 1],
            grad_sqrt_pi: vec![na::SVector::<f64, DIM>::zeros(); 1],
        };
    }

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    // last edge is root edge, we skip it because it doesn't contribute to the gradient
    for edge in 0..tree.parents.len() - 1 {
        crate::backward::d_expm_vjp(
            &mut tls.d_trans[edge],
            tree.distances[edge],
            param,
            &forward_data.model_edge_data[edge].exp_t_lambda,
        );
        d_Q += &tls.d_trans[edge];
    }

    let d_Q = param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose();

    let (grad_s, grad_sqrt_pi) = d_param(d_Q.as_view(), param);

    d_sqrt_pi_sum += grad_sqrt_pi;

    FelsensteinResult {
        log_likelihood: log_likelihoods,
        grad_s: vec![grad_s],
        grad_sqrt_pi: vec![d_sqrt_pi_sum],
    }
}

pub fn calculate_columns_single_S_parallel<const DIM: usize>(
    pl: &mut [Vec<na::SVector<f64, DIM>>],
    S: &na::SMatrix<f64, DIM, DIM>,
    sqrt_pi: &na::SVector<f64, DIM>,
    tree: Tree,
    only_likelihood: bool,
) -> FelsensteinResult<DIM> {
    let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));

    let param = match compute_param_data(S.as_view(), sqrt_pi.as_view()) {
        Some(param) => param,
        None => {
            return FelsensteinResult::<DIM> {
                log_likelihood: vec![f64::NEG_INFINITY; pl.len()],
                grad_s: vec![na::SMatrix::<f64, DIM, DIM>::zeros(); 1],
                grad_sqrt_pi: vec![na::SVector::<f64, DIM>::zeros(); 1],
            };
        }
    };

    let forward_data = forward_data_precompute_param(&param, tree.distances);

    let bifurcations = get_topological_bifurcations(&tree)
        .into_iter()
        .rev()
        .collect::<Vec<_>>();

    let num_threads = rayon::current_num_threads();

    let L = pl.len();

    let real_num_threads = if L / 32 < num_threads {
        L / 32
    } else {
        num_threads
    };

    let base_size = L / real_num_threads;
    let remainder = L % real_num_threads;

    let mut pl = pl;

    let mut results_vec = vec![
        FelsensteinResult::<DIM> {
            log_likelihood: vec![],
            grad_s: vec![],
            grad_sqrt_pi: vec![],
        };
        real_num_threads
    ];

    let mut results = results_vec.as_mut_slice();

    rayon::scope(|s| {
        for i in 0..real_num_threads {
            let end = base_size + if i < remainder { 1 } else { 0 };
            let (first, end) = pl.split_at_mut(end);
            let pl_slice = first;
            pl = end;
            let (first, end) = results.split_first_mut().unwrap();
            results = end;
            s.spawn(|_| {
                *first = calculate_column_block_single_S(
                    pl_slice,
                    &param,
                    &forward_data,
                    tree.clone(),
                    only_likelihood,
                    &bifurcations,
                );
            });
        }
    });

    let log_likelihoods = results_vec
        .iter()
        .flat_map(|res| res.log_likelihood.clone())
        .collect::<Vec<_>>();

    let grad_s = results_vec.iter().map(|res| res.grad_s[0]).sum();
    let grad_sqrt_pi = results_vec.iter().map(|res| res.grad_sqrt_pi[0]).sum();

    FelsensteinResult {
        log_likelihood: log_likelihoods,
        grad_s: vec![grad_s],
        grad_sqrt_pi: vec![grad_sqrt_pi],
    }
}
