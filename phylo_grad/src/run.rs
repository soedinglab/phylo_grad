use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::iter::IntoParallelRefMutIterator;

use crate::backward::{self, *};
use crate::forward::*;
use crate::tree::*;

use nalgebra as na;

/// Sets the interior partial likelihoods (internal_pl, without the leaves) and runs the forward pass.
/// Afterwards the interior nodes except the root hold V_pi_inv * partial likelihood, because this is all the backward pass needs.
fn forward_column<const DIM: usize>(
    leaves: Leaves<DIM>,
    internal_pl: &mut [na::SVector<f64, DIM>],
    lin_child_contributions: &mut [na::SVector<f64, DIM>],
    lin_sibling_prefix: &mut [na::SVector<f64, DIM>],
    parents: &[i32],
    offsets: &mut [u32],
    forward_data: &[ModelEdgeData<DIM>],
    param: &ParamPrecomp<DIM>,
) {
    let num_leaves = leaves.len();
    internal_pl.fill(na::SVector::<f64, DIM>::from_element(1.0));

    for (child, &parent) in parents.iter().enumerate() {
        if parent == -1 {
            continue; // skip the root
        }
        let parent = parent as usize;
        let child_projection = if child < num_leaves {
            match leaves.projection(child, param) {
                Some(projection) => projection,
                None => continue, // gap
            }
        } else {
            let projection = param.V_pi_inv * internal_pl[child - num_leaves];
            internal_pl[child - num_leaves] = projection;
            projection
        };
        forward_node(
            child,
            &child_projection,
            &mut internal_pl[parent - num_leaves],
            &mut offsets[parent],
            lin_child_contributions,
            lin_sibling_prefix,
            forward_data,
            param,
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

/// The buffers of one column, they can be reused for the next column
#[derive(Debug, Clone)]
pub struct ColumnBuffers<const DIM: usize> {
    /// Partial likelihoods of the interior nodes
    internal_pl: Vec<na::SVector<f64, DIM>>,
    offsets: Vec<u32>,
    lin_child_contributions: Vec<na::SVector<f64, DIM>>,
    lin_sibling_prefix: Vec<na::SVector<f64, DIM>>,
}

impl<const DIM: usize> ColumnBuffers<DIM> {
    pub fn new(num_nodes: usize, num_leaves: usize) -> Self {
        ColumnBuffers {
            internal_pl: vec![na::SVector::<f64, DIM>::zeros(); num_nodes - num_leaves],
            offsets: vec![0; num_nodes],
            lin_child_contributions: vec![na::SVector::<f64, DIM>::zeros(); num_nodes],
            lin_sibling_prefix: vec![na::SVector::<f64, DIM>::zeros(); num_nodes],
        }
    }
}

/// All buffers needed to calculate one column with discrete leaves.
/// It can be reused for any number of columns of the same tree, so no allocations are needed per column.
#[derive(Debug)]
pub struct Workspace<const DIM: usize> {
    buffers: ColumnBuffers<DIM>,
    edge_data: Vec<ModelEdgeData<DIM>>,
}

impl<const DIM: usize> Workspace<DIM> {
    pub fn new(num_nodes: usize, num_leaves: usize) -> Self {
        Workspace {
            buffers: ColumnBuffers::new(num_nodes, num_leaves),
            edge_data: Vec::with_capacity(num_nodes),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SingleSideResult<F, const DIM: usize> {
    pub log_likelihood: F,
    pub grad_s: na::SMatrix<F, DIM, DIM>,
    pub grad_sqrt_pi: na::SVector<F, DIM>,
}

/// pl starts with the leaves, everything after them is ignored
pub fn calculate_column<const DIM: usize>(
    pl: &[na::SVector<f64, DIM>],
    S: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    tree: Tree,
    only_likelihood: bool,
    grad_edge_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    calculate_column_leaves(
        Leaves::Vectors(&pl[..tree.num_leaves]),
        S,
        sqrt_pi,
        tree,
        only_likelihood,
        grad_edge_lengths,
    )
}

pub fn calculate_column_leaves<const DIM: usize>(
    leaves: Leaves<DIM>,
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
    calculate_column_with_precompute(
        leaves,
        &mut ColumnBuffers::new(tree.parents.len(), tree.num_leaves),
        &param,
        &edge_data,
        tree,
        only_likelihood,
        grad_edge_lengths,
    )
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

#[derive(Debug, Clone)]
pub struct FelsensteinResultWithSingleTree<const DIM: usize> {
    pub log_likelihood: Vec<f64>,
    pub grad_s: Vec<na::SMatrix<f64, DIM, DIM>>,
    pub grad_sqrt_pi: Vec<na::SVector<f64, DIM>>,
    pub grad_tree: Vec<f64>,
}

pub fn calculate_column_parallel_single_tree_grad<
    const DIM: usize,
    A: AsMut<[na::SVector<f64, DIM>]> + Send,
>(
    leaf_pl: &mut [A],
    S: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    tree: Tree,
) -> FelsensteinResultWithSingleTree<DIM> {
    let col_results = (leaf_pl, S, sqrt_pi)
        .into_par_iter()
        .fold(
            || {
                let tree_grads = vec![0.0; tree.distances.len()];
                FelsensteinResultWithSingleTree {
                    log_likelihood: vec![],
                    grad_s: vec![],
                    grad_sqrt_pi: vec![],
                    grad_tree: tree_grads,
                }
            },
            |mut acc, (leaf_log_p, S, sqrt_pi)| {
                let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
                let result = calculate_column(
                    leaf_log_p.as_mut(),
                    S.as_view(),
                    sqrt_pi.as_view(),
                    tree.clone(),
                    false,
                    Some(&mut acc.grad_tree),
                ); // The clone is shallow, Tree is cheap to clone
                acc.grad_s.push(result.grad_s);
                acc.grad_sqrt_pi.push(result.grad_sqrt_pi);
                acc.log_likelihood.push(result.log_likelihood);
                acc
            },
        )
        .reduce(
            || FelsensteinResultWithSingleTree {
                log_likelihood: vec![],
                grad_s: vec![],
                grad_sqrt_pi: vec![],
                grad_tree: vec![0.0; tree.distances.len()],
            },
            |mut acc1, acc2| {
                acc1.log_likelihood.extend(acc2.log_likelihood);
                acc1.grad_s.extend(acc2.grad_s);
                acc1.grad_sqrt_pi.extend(acc2.grad_sqrt_pi);
                acc1.grad_tree
                    .iter_mut()
                    .zip(acc2.grad_tree.iter())
                    .for_each(|(a, b)| *a += b);
                acc1
            },
        );

    col_results
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
    leaves: Leaves<DIM>,
    // V_pi_inv * partial likelihoods of the interior nodes, from the forward pass
    internal_projections: &[na::SVector<f64, DIM>],
    lin_child_contributions: &[na::SVector<f64, DIM>],
    lin_sibling_prefix: &mut [na::SVector<f64, DIM>],
    param: &ParamPrecomp<DIM>,
    forward: &[ModelEdgeData<DIM>],
    offsets: &[u32],
    mut d_edge_lengths: Option<&mut [f64]>,
) -> na::SMatrix<f64, DIM, DIM> {
    // The cotangents are stored in the sibling prefix buffer: the prefix of a node is read exactly once,
    // right before its cotangent is written. The root has no prefix.
    let cotangents = lin_sibling_prefix;

    // The cotangents are with respect to the partial likelihoods without their offset
    fn offset_scaler(offset: u32) -> f64 {
        if offset == 0 {
            1.0
        } else {
            f64::powi(2.0, offset as i32)
        }
    }

    let num_nodes = tree.parents.len();
    let num_leaves = tree.num_leaves;
    cotangents[num_nodes - 1] = grad_p_root * offset_scaler(offsets[num_nodes - 1]);

    let mut d_Q = na::SMatrix::<f64, DIM, DIM>::zeros();

    // Reverse topological order, so the cotangent of the parent is final before its children are visited
    for child in (0..num_nodes - 1).rev() {
        let parent = tree.parents[child] as usize;

        // The cotangents of the leaves are not needed
        let is_leaf = child < num_leaves;
        let Ainv_x = if is_leaf {
            match leaves.projection(child, param) {
                Some(projection) => projection,
                None => continue, // gap, its contribution is one and has no gradient
            }
        } else {
            internal_projections[child - num_leaves]
        };

        // The parent partial likelihood is the product of the child contributions.
        // The cotangent of one contribution is the parent cotangent times the contributions of all siblings.
        // cotangents[parent] accumulates the siblings after this child, cotangents[child] still holds the prefix of the ones before it.
        let cotangent_vector = cotangents[parent].component_mul(&cotangents[child]);
        cotangents[parent].component_mul_assign(&lin_child_contributions[child]);

        backward::d_edge_vjp(
            &mut d_Q,
            &cotangent_vector,
            tree.distances[child],
            param,
            &forward[child].exp_t_lambda,
            &Ainv_x,
            (!is_leaf).then_some(&mut cotangents[child]),
            d_edge_lengths.as_deref_mut().map(|d| &mut d[child]),
        );
        if !is_leaf && offsets[child] != 0 {
            cotangents[child] *= offset_scaler(offsets[child]);
        }
    }

    param.V_pi_inv.tr_mul(&d_Q) * param.V_pi.transpose()
}

fn calculate_column_with_precompute<const DIM: usize>(
    leaves: Leaves<DIM>,
    buffers: &mut ColumnBuffers<DIM>,
    param: &ParamPrecomp<DIM>,
    forward_data: &[ModelEdgeData<DIM>],
    tree: Tree,
    only_likelihood: bool,
    grad_edge_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    assert_eq!(leaves.len(), tree.num_leaves);
    assert_eq!(buffers.offsets.len(), tree.parents.len());
    assert_eq!(buffers.internal_pl.len(), tree.parents.len() - tree.num_leaves);
    let ColumnBuffers {
        internal_pl,
        offsets,
        lin_child_contributions,
        lin_sibling_prefix,
    } = buffers;
    // Gap edges are skipped and the other entries of the buffers are always written before they are read, only the offsets accumulate
    offsets.fill(0);
    forward_column(
        leaves,
        internal_pl,
        lin_child_contributions,
        lin_sibling_prefix,
        tree.parents,
        offsets,
        forward_data,
        param,
    );

    let lin_pl_root = internal_pl.last().unwrap();

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
        leaves,
        internal_pl,
        lin_child_contributions,
        lin_sibling_prefix,
        param,
        forward_data,
        offsets,
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
                    Leaves::Vectors(&leaf_log_p[..tree.num_leaves]),
                    &mut ColumnBuffers::new(tree.parents.len(), tree.num_leaves),
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
                    Leaves::Vectors(&leaf_log_p[..tree.num_leaves]),
                    &mut ColumnBuffers::new(tree.parents.len(), tree.num_leaves),
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

/// One column with discrete leaves, all buffers are taken from the workspace
pub fn calculate_column_workspace<const DIM: usize>(
    leaves: &[u8],
    S: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    tree: Tree,
    workspace: &mut Workspace<DIM>,
    grad_edge_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
    // If the diagonalization fails or eigenvalues are to big, we give -inf as likelihood and zero gradients
    let param = match compute_param_data(S, sqrt_pi.as_view()) {
        Some(param) => param,
        None => {
            return SingleSideResult::<f64, DIM> {
                log_likelihood: f64::NEG_INFINITY,
                grad_s: na::SMatrix::<f64, DIM, DIM>::zeros(),
                grad_sqrt_pi: na::SVector::<f64, DIM>::zeros(),
            };
        }
    };

    forward_data_precompute_param_into(&param, tree.distances, &mut workspace.edge_data);
    calculate_column_with_precompute(
        Leaves::Discrete(leaves),
        &mut workspace.buffers,
        &param,
        &workspace.edge_data,
        tree,
        false,
        grad_edge_lengths,
    )
}

/// Parallel over the columns with discrete leaves. Every rayon task has its own workspace, which is reused for all its columns.
///
/// S and sqrt_pi have either one entry per column or a single entry for all columns.
/// If grad_branch_lengths has one entry per column, the branch length gradients are added per column.
/// If it has a single entry, the branch length gradients summed over all columns are added to it.
pub fn calculate_columns_discrete_parallel<
    const DIM: usize,
    L: AsRef<[u8]> + Sync,
    G: AsMut<[f64]> + Send,
>(
    leaves: &[L],
    S: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    tree: Tree,
    grad_branch_lengths: Option<&mut [G]>,
) -> FelsensteinResult<DIM> {
    let num_columns = leaves.len();
    let num_nodes = tree.parents.len();
    let single_model = S.len() == 1 && sqrt_pi.len() == 1;
    assert!(
        single_model || (S.len() == num_columns && sqrt_pi.len() == num_columns),
        "S and sqrt_pi need one entry per column or a single entry"
    );

    // Where the branch length gradients of each column go: per column directly into the output,
    // summed into a buffer of the rayon task, or nowhere
    let mut grad_per_column: Vec<Option<&mut [f64]>> = Vec::with_capacity(num_columns);
    let mut grad_sum_output = None;
    match grad_branch_lengths {
        Some(grads) if grads.len() == num_columns => {
            grad_per_column.extend(grads.iter_mut().map(|grad| Some(grad.as_mut())));
        }
        Some(grads) => {
            assert_eq!(
                grads.len(),
                1,
                "grad_branch_lengths needs one entry per column or a single entry"
            );
            grad_sum_output = Some(grads[0].as_mut());
            grad_per_column.resize_with(num_columns, || None);
        }
        None => grad_per_column.resize_with(num_columns, || None),
    }
    let summed = grad_sum_output.is_some();

    // With a single model the eigendecomposition and the edge data are shared by all columns
    let shared_model = if single_model {
        let sqrt_pi = sqrt_pi[0].map(|x| f64::max(x, crate::MIN_SQRT_PI));
        match compute_param_data(S[0].as_view(), sqrt_pi.as_view()) {
            Some(param) => {
                let edge_data = forward_data_precompute_param(&param, tree.distances);
                Some((param, edge_data))
            }
            None => {
                return FelsensteinResult {
                    log_likelihood: vec![f64::NEG_INFINITY; num_columns],
                    grad_s: vec![na::SMatrix::<f64, DIM, DIM>::zeros(); num_columns],
                    grad_sqrt_pi: vec![na::SVector::<f64, DIM>::zeros(); num_columns],
                };
            }
        }
    } else {
        None
    };

    let mut log_likelihood = vec![0.0; num_columns];
    let mut grad_s = vec![na::SMatrix::<f64, DIM, DIM>::zeros(); num_columns];
    let mut grad_sqrt_pi = vec![na::SVector::<f64, DIM>::zeros(); num_columns];

    let grad_sum = (
        log_likelihood.par_iter_mut(),
        grad_s.par_iter_mut(),
        grad_sqrt_pi.par_iter_mut(),
        grad_per_column.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .fold(
            || {
                (
                    Workspace::<DIM>::new(num_nodes, tree.num_leaves),
                    // Empty vectors don't allocate
                    vec![0.0; if summed { num_nodes } else { 0 }],
                )
            },
            |(mut workspace, mut grad_sum),
             (col, (log_likelihood, grad_s, grad_sqrt_pi, grad_column))| {
                let grad = if summed {
                    Some(&mut grad_sum[..])
                } else {
                    grad_column.as_deref_mut()
                };
                let result = match &shared_model {
                    Some((param, edge_data)) => calculate_column_with_precompute(
                        Leaves::Discrete(leaves[col].as_ref()),
                        &mut workspace.buffers,
                        param,
                        edge_data,
                        tree.clone(),
                        false,
                        grad,
                    ),
                    None => calculate_column_workspace(
                        leaves[col].as_ref(),
                        S[col].as_view(),
                        sqrt_pi[col].as_view(),
                        tree.clone(),
                        &mut workspace,
                        grad,
                    ),
                };
                *log_likelihood = result.log_likelihood;
                *grad_s = result.grad_s;
                *grad_sqrt_pi = result.grad_sqrt_pi;
                (workspace, grad_sum)
            },
        )
        .map(|(_, grad_sum)| grad_sum)
        .reduce(
            || vec![0.0; if summed { num_nodes } else { 0 }],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b);
                a
            },
        );

    if let Some(output) = grad_sum_output {
        output.iter_mut().zip(grad_sum.iter()).for_each(|(a, b)| *a += b);
    }

    FelsensteinResult {
        log_likelihood,
        grad_s,
        grad_sqrt_pi,
    }
}
