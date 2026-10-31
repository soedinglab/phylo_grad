#![allow(non_snake_case)]

//! # PhyloGrad
//! This crate provides a Rust implementation of a fast differentiation algorithm for the rate matrix in phylogenetic models.
//! For usage refer to the [FelsensteinTree] struct.
//!
//! If you are looking to use this from Python, you can find information on <https://github.com/soedinglab/phylo_grad>
//!
//! # SIMD
//!
//! This crate uses the `portable_simd` feature to enable SIMD acceleration. This feature is not stable yet, so you need to use the nightly compiler for now.
//! It is tested on `rustc 1.88.0-nightly (7918c7eb5 2025-04-27)`

/// Export the nalgebra which is used in the library, this can enable using multiple versions of nalgebra in the same project
pub use nalgebra;

use nalgebra as na;

mod backward;
mod forward;
mod numerics;
mod run;
mod tree;

pub use run::FelsensteinResult;
pub use run::FelsensteinResultWithSingleTree;
pub use run::FelsensteinResultWithTree;
pub use run::SingleSideResult;
pub use run::Workspace;

use crate::run::*;

const MIN_SQRT_PI: f64 = 1e-10;

/// Represents a tree topology with branch lengths
/// This struct contains the main functionality of the library.
///
/// It is generic over the number of states in the model, which is given by `DIM`.
#[derive(Debug, Clone)]
pub struct FelsensteinTree<const DIM: usize> {
    parents: Vec<i32>,
    distances: Vec<f64>,
    num_leaves: usize,
    /// First dimension is the side id in the alignment, second dimension is the node id in the tree.
    partial_likelihoods: Vec<Vec<na::SVector<f64, DIM>>>,
    /// Sorting order of the nodes. i[new_index] = original_index
    sorting_order: Vec<usize>,
}

impl<const DIM: usize> FelsensteinTree<DIM> {
    /// The tree topology is represented as a vector of parent node ids. The root node has parent id `-1`.
    /// The leaf nodes have to come first in this slice.
    ///
    /// The distances are given as a vector of branch lengths with the same order as the parent vector.
    pub fn new(parents: &[i32], distances: &[f64]) -> Self {
        assert!(parents.len() == distances.len());
        let (parents, distances, num_leaves, sorting_order) =
            tree::topological_sort(parents, distances);

        assert_eq!(parents.last().unwrap(), &-1); // root node is the last node

        FelsensteinTree {
            parents,
            distances,
            partial_likelihoods: vec![],
            num_leaves,
            sorting_order,
        }
    }

    pub fn update_branch_lengths(&mut self, branch_lengths: &[f64]) {
        assert!(branch_lengths.len() == self.distances.len());
        // permute the branch lengths:
        let branch_lengths = {
            let mut permuted_branch_lengths = vec![0.0; branch_lengths.len()];
            for (new, orig) in self.sorting_order.iter().enumerate() {
                permuted_branch_lengths[new] = branch_lengths[*orig];
            }
            permuted_branch_lengths
        };
        self.distances = branch_lengths;
    }

    /// Binds the probabilities of the leaves to the tree. This will usually be a one hot vector describing the state at the leaf node.
    /// The outer vector is over the sites, the inner vector over the leaf nodes.
    /// This enables usage of the `calculate_gradients` function.
    pub fn bind_leaf_pl(&mut self, pl: Vec<Vec<na::SVector<f64, DIM>>>) {
        for column in &pl {
            assert!(column.len() >= self.num_leaves);
        }
        self.partial_likelihoods = pl;
    }

    pub fn num_nodes(&self) -> usize {
        self.parents.len()
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    pub fn num_sites(&self) -> usize {
        self.partial_likelihoods.len()
    }

    /// `s` and `sqrt_pi` have as first dimension the side id in the alignment. `s` gives the state transition matrix for each side, `sqrt_pi` gives the square root of the stationary distribution for each side.
    /// See the paper for more details. Especially the `Time symmetric parameterization` section.
    ///
    /// The result contains the gradients of `s` and `sqrt_pi` with respect to the log likelihood of the tree. It also gives the log likelihood of the tree.
    ///
    /// This function internally parallelizes over the sides in the alignment. You can control the number of threads with the `RAYON_NUM_THREADS` environment variable.
    ///
    /// If the length of `s` and `sqrt_pi` is 1, it will use a different code path that is optimized for this case and assumes that they are the same for all columns.
    ///
    /// Only the upper diagonal part of `s` is used. The gradients will only be populated in the upper diagonal and the lower diagonal will be filled with zeros.
    ///
    /// This functions assumes you have already called `bind_leaf_pl` to bind the partial likelihoods of the leaves.
    pub fn calculate_gradients(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
    ) -> FelsensteinResult<DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);

        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            calculate_columns_single_S_parallel(
                &mut self.partial_likelihoods,
                &s[0],
                &sqrt_pi[0],
                tree,
                false,
                None,
            )
        } else {
            let mut tree_grad = None;
            calculate_column_parallel(
                &mut self.partial_likelihoods,
                s,
                sqrt_pi,
                tree,
                false,
                &mut tree_grad,
            )
        };
        result
    }

    pub fn calculate_likelihoods_with_branch_lengths(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
        branch_lengths: &[f64],
    ) -> Vec<f64> {
        // permute the branch lengths:
        let branch_lengths = {
            let mut permuted_branch_lengths = vec![0.0; branch_lengths.len()];
            for (new, orig) in self.sorting_order.iter().enumerate() {
                permuted_branch_lengths[new] = branch_lengths[*orig];
            }
            permuted_branch_lengths
        };

        let tree = tree::Tree::new(&self.parents, &branch_lengths, self.num_leaves);


        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            calculate_columns_single_S_parallel(
                &mut self.partial_likelihoods,
                &s[0],
                &sqrt_pi[0],
                tree,
                true,
                None,
            )
        } else {
            let mut tree_grad = None;
            calculate_column_parallel(
                &mut self.partial_likelihoods,
                s,
                sqrt_pi,
                tree,
                true,
                &mut tree_grad,
            )
        };

        result.log_likelihood
    }

    pub fn calculate_gradients_with_branch_lengths(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
        branch_lengths: &[f64],
    ) -> FelsensteinResultWithTree<DIM> {
        // permute the branch lengths:
        let branch_lengths = {
            let mut permuted_branch_lengths = vec![0.0; branch_lengths.len()];
            for (new, orig) in self.sorting_order.iter().enumerate() {
                permuted_branch_lengths[new] = branch_lengths[*orig];
            }
            permuted_branch_lengths
        };

        let tree = tree::Tree::new(&self.parents, &branch_lengths, self.num_leaves);

        let mut branch_length_grads =
            vec![vec![0.0; self.parents.len()]; self.partial_likelihoods.len()];

        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            calculate_columns_single_S_parallel(
                &mut self.partial_likelihoods,
                &s[0],
                &sqrt_pi[0],
                tree,
                false,
                Some(&mut branch_length_grads),
            )
        } else {
            let mut tree_grad = Some(branch_length_grads);
            let result = calculate_column_parallel(
                &mut self.partial_likelihoods,
                s,
                sqrt_pi,
                tree,
                false,
                &mut tree_grad,
            );
            branch_length_grads = tree_grad.unwrap();
            result
        };

        // Permute back the branch length gradients:
        let branch_length_grads = {
            let mut permuted_branch_length_grads =
                vec![vec![0.0; self.parents.len()]; branch_length_grads.len()];
            for (col, grad_col) in branch_length_grads.into_iter().enumerate() {
                for (new, orig) in self.sorting_order.iter().enumerate() {
                    permuted_branch_length_grads[col][*orig] = grad_col[new];
                }
            }
            permuted_branch_length_grads
        };

        FelsensteinResultWithTree {
            log_likelihood: result.log_likelihood,
            grad_s: result.grad_s,
            grad_sqrt_pi: result.grad_sqrt_pi,
            grad_tree: branch_length_grads,
        }
    }

    /// Same as calculate_gradients_with_branch_lengths, but the gradient of the branch lengths is calculated as the sum of the log likelihoods over all sites in the alignment.
    /// This is the same as summing up all the gradients of calculate_gradients_with_branch_lengths but it's faster.
    pub fn calculate_gradients_with_branch_lengths_sum(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
        branch_lengths: &[f64],
    ) -> FelsensteinResultWithSingleTree<DIM> {
        // permute the branch lengths:
        let branch_lengths = {
            let mut permuted_branch_lengths = vec![0.0; branch_lengths.len()];
            for (new, orig) in self.sorting_order.iter().enumerate() {
                permuted_branch_lengths[new] = branch_lengths[*orig];
            }
            permuted_branch_lengths
        };

        let tree = tree::Tree::new(&self.parents, &branch_lengths, self.num_leaves);

        let mut result = calculate_column_parallel_single_tree_grad(
            &mut self.partial_likelihoods,
            s,
            sqrt_pi,
            tree,
        );

        // Permute back the branch length gradients:
        let branch_length_grads = {
            let mut permuted_branch_length_grads = vec![0.0; self.parents.len()];
            for (new, orig) in self.sorting_order.iter().enumerate() {
                permuted_branch_length_grads[*orig] = result.grad_tree[new];
            }
            permuted_branch_length_grads
        };

        result.grad_tree = branch_length_grads;

        result
    }
    /// Same as `calculate_gradients`, but only calculates the log likelihoods for each side in the alignment.
    pub fn calculate_likelihoods(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
    ) -> Vec<f64> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);

        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            calculate_columns_single_S_parallel(
                &mut self.partial_likelihoods,
                &s[0],
                &sqrt_pi[0],
                tree,
                true,
                None,
            )
        } else {
            let mut tree_grad = None;
            calculate_column_parallel(
                &mut self.partial_likelihoods,
                s,
                sqrt_pi,
                tree,
                true,
                &mut tree_grad,
            )
        };

        return result.log_likelihood;
    }

    /// Same as `calculate_gradients`, but it takes also an array of the partial likelihoods of the leaves.
    /// Only the leaf nodes at the start of each column of `pl` are read, everything after them is ignored.
    pub fn calculate_gradients_with_pl(
        &self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
        partial_likelihood: &mut [&mut [na::SVector<f64, DIM>]],
    ) -> FelsensteinResult<DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        let mut tree_grad = None;
        calculate_column_parallel(partial_likelihood, s, sqrt_pi, tree, false, &mut tree_grad)
    }

    /// This function calculates the gradients for a single side in the alignment.
    /// This can be useful if you want to control the parallelization yourself or if you want to calculate the gradients for a single side.
    ///
    /// Only the leaf nodes at the start of `partial_likelihood` are read, everything after them is ignored.
    pub fn calculate_gradients_single_side(
        &self,
        s: na::SMatrixView<f64, DIM, DIM>,
        sqrt_pi: na::SVectorView<f64, DIM>,
        partial_likelihood: &mut [na::SVector<f64, DIM>],
    ) -> SingleSideResult<f64, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
        calculate_column(
            partial_likelihood,
            s.as_view(),
            sqrt_pi.as_view(),
            tree,
            false,
            None,
        )
    }

    /// Same as `calculate_gradients_single_side`, but only calculates the log likelihood for a single side in the alignment.
    pub fn calculate_likelihood_single_side(
        &self,
        s: na::SMatrixView<f64, DIM, DIM>,
        sqrt_pi: na::SVectorView<f64, DIM>,
        partial_likelihood: &mut [na::SVector<f64, DIM>],
    ) -> f64 {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        let sqrt_pi = sqrt_pi.map(|x| f64::max(x, crate::MIN_SQRT_PI));
        let result = calculate_column(
            partial_likelihood,
            s.as_view(),
            sqrt_pi.as_view(),
            tree,
            true,
            None,
        );
        result.log_likelihood
    }
}
/// Checks the input of [calculate_gradients_single_column] and [calculate_gradients_columns]
/// Checks the input of [calculate_gradients_single_column] and [calculate_gradients_columns]
fn check_sorted_tree(parents: &[i32], num_leaves: usize, branch_lengths: &[f64]) {
    let num_nodes = parents.len();
    assert_eq!(branch_lengths.len(), num_nodes);
    assert!(num_leaves < num_nodes);
    assert_eq!(parents.last(), Some(&-1)); // root node is the last node
    assert!(
        parents[..num_nodes - 1]
            .iter()
            .enumerate()
            .all(|(child, &parent)| parent as usize > child && (parent as usize) < num_nodes),
        "parents are not topologically sorted"
    );
}

fn check_discrete_leaves<const DIM: usize>(leaves: &[u8], num_leaves: usize) {
    assert_eq!(leaves.len(), num_leaves);
    assert!(
        leaves.iter().all(|&state| state as usize <= DIM),
        "leaf states have to be in 0..={}",
        DIM
    );
}

/// Calculates the log likelihood and the gradients for a single column without building a [FelsensteinTree].
///
/// `parents` has to be topologically sorted already: the leaves are the first `leaves.len()` nodes,
/// the parent of every node has a bigger index than the node itself and the root is the last node with parent `-1`.
/// Nodes can have any number of children. The input is only checked in debug builds.
///
/// `leaves` gives the state of each leaf as an integer in `0..DIM`, `DIM` stands for a gap (all states are possible).
/// Gaps cost nothing, a leaf with a state is cheaper than a leaf partial likelihood vector.
///
/// `branch_lengths` has the same order as `parents`, the entry of the root is ignored.
///
/// `workspace` holds all buffers, create it with `Workspace::new(parents.len(), leaves.len())` and reuse it for all columns,
/// then nothing is allocated.
///
/// If `grad_branch_lengths` is given, the gradients of the branch lengths are added to it (it is not zeroed out).
pub fn calculate_gradients_single_column<const DIM: usize>(
    parents: &[i32],
    leaves: &[u8],
    branch_lengths: &[f64],
    s: na::SMatrixView<f64, DIM, DIM>,
    sqrt_pi: na::SVectorView<f64, DIM>,
    workspace: &mut Workspace<DIM>,
    grad_branch_lengths: Option<&mut [f64]>,
) -> SingleSideResult<f64, DIM> {
    let num_leaves = leaves.len();
    if cfg!(debug_assertions) {
        check_sorted_tree(parents, num_leaves, branch_lengths);
        check_discrete_leaves::<DIM>(leaves, num_leaves);
        if let Some(grad) = grad_branch_lengths.as_deref() {
            assert_eq!(grad.len(), parents.len());
        }
    }

    let tree = tree::Tree::new(parents, branch_lengths, num_leaves);
    calculate_column_workspace(leaves, s, sqrt_pi, tree, workspace, grad_branch_lengths)
}

/// Parallel version of [calculate_gradients_single_column] over the columns of an alignment.
///
/// `leaves[col]` holds the states of the leaves in column `col`, with the same conventions as in [calculate_gradients_single_column].
///
/// `s` and `sqrt_pi` have one entry per column, or a single entry which is used for all columns.
/// With a single entry, the eigendecomposition is only done once.
///
/// If `grad_branch_lengths` has one entry per column, the branch length gradients of each column are added to its entry.
/// If it has a single entry, the gradients summed over all columns are added to it. The entries are in the order of `parents`.
///
/// Every rayon task allocates its buffers once and reuses them for all its columns.
/// You can control the number of threads with the `RAYON_NUM_THREADS` environment variable.
pub fn calculate_gradients_columns<
    const DIM: usize,
    L: AsRef<[u8]> + Sync,
    G: AsMut<[f64]> + Send,
>(
    parents: &[i32],
    leaves: &[L],
    branch_lengths: &[f64],
    s: &[na::SMatrix<f64, DIM, DIM>],
    sqrt_pi: &[na::SVector<f64, DIM>],
    grad_branch_lengths: Option<&mut [G]>,
) -> FelsensteinResult<DIM> {
    let num_leaves = leaves.first().map_or(0, |col| col.as_ref().len());
    if cfg!(debug_assertions) {
        check_sorted_tree(parents, num_leaves, branch_lengths);
        for col in leaves {
            check_discrete_leaves::<DIM>(col.as_ref(), num_leaves);
        }
    }
    let mut grad_branch_lengths = grad_branch_lengths;
    if cfg!(debug_assertions) {
        for grad in grad_branch_lengths.iter_mut().flat_map(|grads| grads.iter_mut()) {
            assert_eq!(grad.as_mut().len(), parents.len());
        }
    }

    let tree = tree::Tree::new(parents, branch_lengths, num_leaves);
    calculate_columns_discrete_parallel(leaves, s, sqrt_pi, tree, grad_branch_lengths)
}

// Unit tests:
#[cfg(test)]
mod tests {
    use rand::{SeedableRng, distr::Distribution};

    use super::*;

    fn random_S(rng: &mut impl rand::Rng) -> na::SMatrix<f64, 4, 4> {
        let mut m = na::SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            for j in (i + 1)..4 {
                m[(i, j)] = rand::distr::Uniform::new(0.1, 1.0).unwrap().sample(rng);
            }
        }
        m
    }

    fn random_sqrt_pi(rng: &mut impl rand::Rng) -> na::SVector<f64, 4> {
        let mut v = na::SVector::<f64, 4>::zeros();
        for i in 0..4 {
            v[i] = rand::distr::Uniform::new(0.1, 1.0).unwrap().sample(rng);
        }
        v /= v.sum();
        v.map(|x| x.sqrt())
    }

    fn random_pl(rng: &mut impl rand::Rng, num_leaves: usize) -> Vec<na::SVector<f64, 4>> {
        (0..num_leaves)
            .map(|_| {
                let mut v = na::SVector::<f64, 4>::zeros();
                v[rand::distr::Uniform::new(0, 4).unwrap().sample(rng)] = 1.0;
                v
            })
            .collect()
    }

    fn random_branch_lengths(rng: &mut impl rand::Rng, num_nodes: usize) -> Vec<f64> {
        rand::distr::Uniform::new(0.1, 1.0)
            .unwrap()
            .sample_iter(rng)
            .take(num_nodes)
            .collect()
    }

    fn numerical_grads(
        tree: &mut FelsensteinTree<4>,
        s: &mut [na::SMatrix<f64, 4, 4>],
        sqrt_pi: &mut [na::SVector<f64, 4>],
        epsilon: f64,
    ) -> FelsensteinResult<4> {
        let mut grad_s = vec![na::SMatrix::<f64, 4, 4>::zeros(); s.len()];
        let mut grad_sqrt_pi = vec![na::SVector::<f64, 4>::zeros(); sqrt_pi.len()];

        for i in 0..s.len() {
            for j in 0..4 {
                for k in 0..4 {
                    let original_value = s[i][(j, k)];
                    s[i][(j, k)] = original_value + epsilon;
                    let plus_epsilon = tree.calculate_likelihoods(s, sqrt_pi)[i];
                    s[i][(j, k)] = original_value - epsilon;
                    let minus_epsilon = tree.calculate_likelihoods(s, sqrt_pi)[i];
                    grad_s[i][(j, k)] = (plus_epsilon - minus_epsilon) / (2.0 * epsilon);
                    s[i][(j, k)] = original_value; // restore original value
                }
            }
        }

        for i in 0..sqrt_pi.len() {
            for j in 0..4 {
                let original_value = sqrt_pi[i][j];
                sqrt_pi[i][j] = original_value + epsilon;
                let plus_epsilon = tree.calculate_likelihoods(s, sqrt_pi)[i];
                sqrt_pi[i][j] = original_value - epsilon;
                let minus_epsilon = tree.calculate_likelihoods(s, sqrt_pi)[i];
                grad_sqrt_pi[i][j] = (plus_epsilon - minus_epsilon) / (2.0 * epsilon);
                sqrt_pi[i][j] = original_value; // restore original value
            }
        }

        FelsensteinResult {
            log_likelihood: tree.calculate_likelihoods(s, sqrt_pi),
            grad_s,
            grad_sqrt_pi,
        }
    }

    fn numerical_grads_branches(
        parents: &[i32],
        pl: Vec<Vec<na::SVector<f64, 4>>>,
        s: &[na::SMatrix<f64, 4, 4>],
        sqrt_pi: &[na::SVector<f64, 4>],
        branch_lengths: &mut [f64],
        epsilon: f64,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let L = pl.len();
        let mut grad_tree = vec![vec![0.0; branch_lengths.len()]; L];

        let likelihoods = {
            let mut tree = FelsensteinTree::<4>::new(parents, branch_lengths);
            tree.bind_leaf_pl(pl.clone());
            tree.calculate_likelihoods(s, sqrt_pi)
        };

        for i in 0..branch_lengths.len() {
            let original_value = branch_lengths[i];
            branch_lengths[i] = original_value + epsilon;
            let plus_epsilon = {
                let mut tree = FelsensteinTree::<4>::new(parents, branch_lengths);
                tree.bind_leaf_pl(pl.clone());
                tree.calculate_likelihoods(s, sqrt_pi)
            };
            branch_lengths[i] = original_value - epsilon;
            let minus_epsilon = {
                let mut tree = FelsensteinTree::<4>::new(parents, branch_lengths);
                tree.bind_leaf_pl(pl.clone());
                tree.calculate_likelihoods(s, sqrt_pi)
            };
            for col in 0..L {
                grad_tree[col][i] = (plus_epsilon[col] - minus_epsilon[col]) / (2.0 * epsilon);
            }
            branch_lengths[i] = original_value; // restore original value
        }

        (grad_tree, likelihoods)
    }

    #[test]
    fn test_felsenstein_tree() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];

        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;

        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();

        tree.bind_leaf_pl(pl);

        let S = (0..L)
            .map(|_| random_S(&mut rng))
            .collect::<Vec<na::SMatrix<f64, 4, 4>>>();

        let sqrt_pi = (0..L)
            .map(|_| random_sqrt_pi(&mut rng))
            .collect::<Vec<na::SVector<f64, 4>>>();

        let result = tree.calculate_gradients(&S, &sqrt_pi);
        println!("Log likelihoods: {:?}", result.log_likelihood);
        println!("Grad s: {:?}", result.grad_s);
        println!("Grad sqrt_pi: {:?}", result.grad_sqrt_pi);

        let numerical_result =
            numerical_grads(&mut tree, &mut S.clone(), &mut sqrt_pi.clone(), 1e-5);
        println!(
            "Numerical Log likelihoods: {:?}",
            numerical_result.log_likelihood
        );
        println!("Numerical Grad s: {:?}", numerical_result.grad_s);
        println!(
            "Numerical Grad sqrt_pi: {:?}",
            numerical_result.grad_sqrt_pi
        );

        for i in 0..S.len() {
            for j in 0..4 {
                for k in 0..4 {
                    let grad = result.grad_s[i][(j, k)];
                    let numerical_grad = numerical_result.grad_s[i][(j, k)];
                    assert!(
                        (grad - numerical_grad).abs() < 1e-3,
                        "Grad s at ({}, {}, {}) differs: {} vs {}",
                        i,
                        j,
                        k,
                        grad,
                        numerical_grad
                    );
                }
            }
        }

        for i in 0..sqrt_pi.len() {
            for j in 0..4 {
                let grad = result.grad_sqrt_pi[i][j];
                let numerical_grad = numerical_result.grad_sqrt_pi[i][j];
                assert!(
                    (grad - numerical_grad).abs() < 1e-3,
                    "Grad sqrt_pi at ({}, {}) differs: {} vs {}",
                    i,
                    j,
                    grad,
                    numerical_grad
                );
            }
        }
    }
    #[test]
    fn test_felsenstein_tree_branch_grads_rooted() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, 10, 6, 6, 10, -1];

        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;

        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();

        tree.bind_leaf_pl(pl.clone());

        let S = (0..L)
            .map(|_| random_S(&mut rng))
            .collect::<Vec<na::SMatrix<f64, 4, 4>>>();

        let sqrt_pi = (0..L)
            .map(|_| random_sqrt_pi(&mut rng))
            .collect::<Vec<na::SVector<f64, 4>>>();

        let distances = random_branch_lengths(&mut rng, parents.len());

        let result = tree.calculate_gradients_with_branch_lengths(&S, &sqrt_pi, &distances);
        println!("Log likelihoods: {:?}", result.log_likelihood);
        println!("Grad tree: {:?}", result.grad_tree);

        let numerical_result = numerical_grads_branches(
            &parents,
            pl.clone(),
            &S.clone(),
            &sqrt_pi.clone(),
            &mut distances.clone(),
            1e-5,
        );
        println!("Numerical Grad tree: {:?}", numerical_result.0);
        println!("Numerical Log likelihoods: {:?}", numerical_result.1);

        for col in 0..sqrt_pi.len() {
            let grad = &result.grad_tree[col];
            for i in 0..parents.len() {
                let grad = grad[i];
                let numerical_grad = numerical_result.0[col][i];
                assert!(
                    (grad - numerical_grad).abs() < 1e-3,
                    "Grad tree at {} differs: {} vs {}",
                    i,
                    grad,
                    numerical_grad
                );
            }
        }
    }

    #[test]
    fn test_felsenstein_tree_branch_grads() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];

        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;

        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();

        tree.bind_leaf_pl(pl.clone());

        let S = (0..L)
            .map(|_| random_S(&mut rng))
            .collect::<Vec<na::SMatrix<f64, 4, 4>>>();

        let sqrt_pi = (0..L)
            .map(|_| random_sqrt_pi(&mut rng))
            .collect::<Vec<na::SVector<f64, 4>>>();

        let distances = random_branch_lengths(&mut rng, parents.len());

        let result = tree.calculate_gradients_with_branch_lengths(&S, &sqrt_pi, &distances);
        println!("Log likelihoods: {:?}", result.log_likelihood);
        println!("Grad tree: {:?}", result.grad_tree);

        let numerical_result = numerical_grads_branches(
            &parents,
            pl.clone(),
            &S.clone(),
            &sqrt_pi.clone(),
            &mut distances.clone(),
            1e-5,
        );
        println!("Numerical Grad tree: {:?}", numerical_result.0);
        println!("Numerical Log likelihoods: {:?}", numerical_result.1);

        for col in 0..sqrt_pi.len() {
            let grad = &result.grad_tree[col];
            for i in 0..parents.len() {
                let grad = grad[i];
                let numerical_grad = numerical_result.0[col][i];
                assert!(
                    (grad - numerical_grad).abs() < 1e-3,
                    "Grad tree at {} differs: {} vs {}",
                    i,
                    grad,
                    numerical_grad
                );
            }
        }
    }
    #[test]
    fn test_felsenstein_tree_branch_grads_singleS() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];

        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;

        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();

        tree.bind_leaf_pl(pl.clone());

        let S = random_S(&mut rng);

        let sqrt_pi = random_sqrt_pi(&mut rng);

        let distances = random_branch_lengths(&mut rng, parents.len());

        let result =
            tree.calculate_gradients_with_branch_lengths(&vec![S], &vec![sqrt_pi], &distances);
        println!("Log likelihoods: {:?}", result.log_likelihood);
        println!("Grad tree: {:?}", result.grad_tree);
        assert_eq!(result.grad_tree.len(), L);

        let numerical_result = numerical_grads_branches(
            &parents,
            pl.clone(),
            &vec![S],
            &vec![sqrt_pi],
            &mut distances.clone(),
            1e-5,
        );
        println!("Numerical Grad tree: {:?}", numerical_result.0);
        println!("Numerical Log likelihoods: {:?}", numerical_result.1);

        for col in 0..sqrt_pi.len() {
            let grad = &result.grad_tree[col];
            for i in 0..parents.len() {
                let grad = grad[i];
                let numerical_grad = numerical_result.0[col][i];
                assert!(
                    (grad - numerical_grad).abs() < 1e-3,
                    "Grad tree at {} differs: {} vs {}",
                    i,
                    grad,
                    numerical_grad
                );
            }
        }
    }

    #[test]
    fn test_likelihoods_with_branch_lengths() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];

        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;

        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();

        tree.bind_leaf_pl(pl.clone());

        let S = random_S(&mut rng);

        let sqrt_pi = random_sqrt_pi(&mut rng);

        let distances2 = random_branch_lengths(&mut rng, parents.len());

        let result_with_grad =
            tree.calculate_gradients_with_branch_lengths(&vec![S], &vec![sqrt_pi], &distances2);

        let result_without_grad =
            tree.calculate_likelihoods_with_branch_lengths(&vec![S], &vec![sqrt_pi], &distances2);

        println!(
            "Log likelihoods with grad: {:?}",
            result_with_grad.log_likelihood
        );
        println!("Log likelihoods without grad: {:?}", result_without_grad);

        for (l1, l2) in result_with_grad
            .log_likelihood
            .iter()
            .zip(result_without_grad.iter())
        {
            assert!(
                (l1 - l2).abs() < 1e-6,
                "Log likelihoods differ: {} vs {}",
                l1,
                l2
            );
        }
    }

    #[test]
    fn test_branch_grads_sum_ll_matches_per_site_sum() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];
        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 5;
        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();
        tree.bind_leaf_pl(pl);

        let s = (0..L)
            .map(|_| random_S(&mut rng))
            .collect::<Vec<na::SMatrix<f64, 4, 4>>>();
        let sqrt_pi = (0..L)
            .map(|_| random_sqrt_pi(&mut rng))
            .collect::<Vec<na::SVector<f64, 4>>>();
        let branch_lengths = random_branch_lengths(&mut rng, parents.len());

        let with_tree = tree.calculate_gradients_with_branch_lengths(&s, &sqrt_pi, &branch_lengths);
        let with_tree_sum =
            tree.calculate_gradients_with_branch_lengths_sum(&s, &sqrt_pi, &branch_lengths);

        let mut summed_per_site_tree_grad = vec![0.0; parents.len()];
        for col_grad in &with_tree.grad_tree {
            for (i, g) in col_grad.iter().enumerate() {
                summed_per_site_tree_grad[i] += g;
            }
        }

        for (ll_a, ll_b) in with_tree
            .log_likelihood
            .iter()
            .zip(with_tree_sum.log_likelihood.iter())
        {
            assert!(
                (ll_a - ll_b).abs() < 1e-10,
                "Log likelihoods differ: {} vs {}",
                ll_a,
                ll_b
            );
        }

        for (grad_a, grad_b) in with_tree.grad_s.iter().zip(with_tree_sum.grad_s.iter()) {
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (grad_a[(i, j)] - grad_b[(i, j)]).abs() < 1e-10,
                        "Grad s differs at ({}, {}): {} vs {}",
                        i,
                        j,
                        grad_a[(i, j)],
                        grad_b[(i, j)]
                    );
                }
            }
        }

        for (grad_a, grad_b) in with_tree
            .grad_sqrt_pi
            .iter()
            .zip(with_tree_sum.grad_sqrt_pi.iter())
        {
            for i in 0..4 {
                assert!(
                    (grad_a[i] - grad_b[i]).abs() < 1e-10,
                    "Grad sqrt_pi differs at {}: {} vs {}",
                    i,
                    grad_a[i],
                    grad_b[i]
                );
            }
        }

        for (i, summed_grad) in summed_per_site_tree_grad.iter().enumerate() {
            let sum_ll_grad = with_tree_sum.grad_tree[i];
            assert!(
                (summed_grad - sum_ll_grad).abs() < 1e-10,
                "Summed branch grad differs at {}: {} vs {}",
                i,
                summed_grad,
                sum_ll_grad
            );
        }
    }

    #[test]
    fn test_multifurcations() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        // Node 7 has 4 children, node 8 a single child
        let parents = vec![7, 7, 7, 7, 8, 9, 9, 9, 9, -1];
        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 3;
        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();
        tree.bind_leaf_pl(pl.clone());

        let S = (0..L)
            .map(|_| random_S(&mut rng))
            .collect::<Vec<na::SMatrix<f64, 4, 4>>>();
        let sqrt_pi = (0..L)
            .map(|_| random_sqrt_pi(&mut rng))
            .collect::<Vec<na::SVector<f64, 4>>>();

        let result = tree.calculate_gradients_with_branch_lengths(&S, &sqrt_pi, &distances);
        let numerical =
            numerical_grads(&mut tree, &mut S.clone(), &mut sqrt_pi.clone(), 1e-5);
        let (numerical_tree, _) = numerical_grads_branches(
            &parents,
            pl,
            &S,
            &sqrt_pi,
            &mut distances.clone(),
            1e-5,
        );

        for col in 0..L {
            assert!((result.grad_s[col] - numerical.grad_s[col]).abs().max() < 1e-3);
            assert!((result.grad_sqrt_pi[col] - numerical.grad_sqrt_pi[col]).abs().max() < 1e-3);
            for i in 0..parents.len() {
                let (grad, numerical_grad) = (result.grad_tree[col][i], numerical_tree[col][i]);
                assert!(
                    (grad - numerical_grad).abs() < 1e-3,
                    "Grad tree at {} differs: {} vs {}",
                    i,
                    grad,
                    numerical_grad
                );
            }
        }
    }

    #[test]
    fn test_columns_parallel_matches_single_column() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![6, 6, 7, 7, 8, 8, 9, 9, 9, -1];
        let branch_lengths = random_branch_lengths(&mut rng, parents.len());
        let L = 50;
        // Different gap patterns per column, a reused workspace must not leak anything between columns
        let leaves = (0..L)
            .map(|_| {
                (0..6)
                    .map(|_| rand::distr::Uniform::new(0, 5).unwrap().sample(&mut rng))
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<_>>();
        let s = (0..L).map(|_| random_S(&mut rng)).collect::<Vec<_>>();
        let sqrt_pi = (0..L).map(|_| random_sqrt_pi(&mut rng)).collect::<Vec<_>>();

        for single_model in [false, true] {
            let (s, sqrt_pi) = if single_model {
                (&s[..1], &sqrt_pi[..1])
            } else {
                (&s[..], &sqrt_pi[..])
            };

            // The expected values column by column, with one reused workspace
            let mut workspace = Workspace::<4>::new(parents.len(), 6);
            let mut expected_grads = vec![vec![0.0; parents.len()]; L];
            let expected = (0..L)
                .map(|col| {
                    let model = if single_model { 0 } else { col };
                    calculate_gradients_single_column(
                        &parents,
                        &leaves[col],
                        &branch_lengths,
                        s[model].as_view(),
                        sqrt_pi[model].as_view(),
                        &mut workspace,
                        Some(&mut expected_grads[col]),
                    )
                })
                .collect::<Vec<_>>();
            let mut expected_sum = vec![0.0; parents.len()];
            for grad in &expected_grads {
                expected_sum.iter_mut().zip(grad).for_each(|(a, b)| *a += b);
            }

            let mut grads_per_column = vec![vec![0.0; parents.len()]; L];
            // Starts at one, the gradients are added
            let mut grads_sum = vec![vec![1.0; parents.len()]];
            for grads in [None, Some(&mut grads_per_column), Some(&mut grads_sum)] {
                let result = calculate_gradients_columns(
                    &parents,
                    &leaves,
                    &branch_lengths,
                    s,
                    sqrt_pi,
                    grads.map(|grads| &mut grads[..]),
                );
                for col in 0..L {
                    assert_eq!(result.log_likelihood[col], expected[col].log_likelihood);
                    assert_eq!(result.grad_s[col], expected[col].grad_s);
                    assert_eq!(result.grad_sqrt_pi[col], expected[col].grad_sqrt_pi);
                }
            }

            assert_eq!(grads_per_column, expected_grads);
            for (a, b) in grads_sum[0].iter().zip(expected_sum.iter()) {
                assert!((a - (b + 1.0)).abs() < 1e-10, "Branch grad differs: {} vs {}", a, b + 1.0);
            }
        }
    }

    #[test]
    fn test_single_column_matches_tree() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        // Already topologically sorted, trifurcation at the root
        let parents = vec![6, 6, 7, 7, 8, 8, 9, 9, 9, -1];
        let branch_lengths = random_branch_lengths(&mut rng, parents.len());
        let leaves: Vec<u8> = vec![0, 3, 4, 1, 2, 4]; // 4 is a gap

        let leaf_pl = leaves
            .iter()
            .map(|&state| {
                if state == 4 {
                    na::SVector::<f64, 4>::from_element(1.0)
                } else {
                    let mut v = na::SVector::<f64, 4>::zeros();
                    v[state as usize] = 1.0;
                    v
                }
            })
            .collect::<Vec<_>>();

        let s = random_S(&mut rng);
        let sqrt_pi = random_sqrt_pi(&mut rng);

        let mut tree = FelsensteinTree::<4>::new(&parents, &branch_lengths);
        tree.bind_leaf_pl(vec![leaf_pl]);
        let expected =
            tree.calculate_gradients_with_branch_lengths(&[s], &[sqrt_pi], &branch_lengths);

        let mut workspace = Workspace::<4>::new(parents.len(), leaves.len());
        let mut grad_branch_lengths = vec![0.0; parents.len()];
        let result = calculate_gradients_single_column(
            &parents,
            &leaves,
            &branch_lengths,
            s.as_view(),
            sqrt_pi.as_view(),
            &mut workspace,
            Some(&mut grad_branch_lengths),
        );

        assert!((result.log_likelihood - expected.log_likelihood[0]).abs() < 1e-10);
        assert!((result.grad_s - expected.grad_s[0]).abs().max() < 1e-10);
        assert!((result.grad_sqrt_pi - expected.grad_sqrt_pi[0]).abs().max() < 1e-10);
        for (a, b) in grad_branch_lengths.iter().zip(expected.grad_tree[0].iter()) {
            assert!((a - b).abs() < 1e-10, "Branch grad differs: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_update_branch_lengths_same_values_same_likelihood() {
        let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42);

        let parents = vec![7, 7, 8, 8, 9, 9, -1, 6, 6, 6];
        let distances = random_branch_lengths(&mut rng, parents.len());
        let mut tree = FelsensteinTree::<4>::new(&parents, &distances);

        let L = 4;
        let pl = (0..L)
            .map(|_| random_pl(&mut rng, tree.num_leaves()))
            .collect::<Vec<Vec<na::SVector<f64, 4>>>>();
        tree.bind_leaf_pl(pl);

        let s = random_S(&mut rng);
        let sqrt_pi = random_sqrt_pi(&mut rng);

        let likelihood_before = tree.calculate_likelihoods(&vec![s], &vec![sqrt_pi]);
        tree.update_branch_lengths(&distances);
        let likelihood_after = tree.calculate_likelihoods(&vec![s], &vec![sqrt_pi]);

        for (before, after) in likelihood_before.iter().zip(likelihood_after.iter()) {
            assert!(
                (before - after).abs() < 1e-12,
                "Log likelihood changed after noop branch update: {} vs {}",
                before,
                after
            );
        }
    }
}
