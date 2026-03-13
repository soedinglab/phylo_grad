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
pub use run::FelsensteinResultWithTree;
pub use run::SingleSideResult;

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

    /// Binds the probabilities of the leaves to the tree. This will usually be a one hot vector describing the state at the leaf node.
    /// The outer vector is over the sites, the inner vector over the leaf nodes.
    /// This enables usage of the `calculate_gradients` function.
    pub fn bind_leaf_pl(&mut self, pl: Vec<Vec<na::SVector<f64, DIM>>>) {
        self.partial_likelihoods = pl;

        // resize the partial_likelihoods to the number of all nodes
        let num_nodes = self.parents.len();
        for pl in &mut self.partial_likelihoods {
            pl.resize(num_nodes, na::SVector::<f64, DIM>::zeros());
        }
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
        // One out internal nodes in partial_likelihoods
        for pl in &mut self.partial_likelihoods {
            pl.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
        }

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
        // One out internal nodes in partial_likelihoods
        for pl in &mut self.partial_likelihoods {
            pl.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
        }

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

    /// Same as `calculate_gradients`, but only calculates the log likelihoods for each side in the alignment.
    pub fn calculate_likelihoods(
        &mut self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
    ) -> Vec<f64> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // One out internal nodes in partial_likelihoods
        for pl in &mut self.partial_likelihoods {
            pl.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
        }

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
    /// It expects `pl` to have enough space for all nodes with internal nodes initialized to one and leaf nodes properly initialized.
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
    /// partial_likelihood is expected to have enough space to hold the partial likelihoods for all nodes
    /// internal nodes will be initialized to one and leaf nodes should be properly initialized.
    pub fn calculate_gradients_single_side(
        &self,
        s: na::SMatrixView<f64, DIM, DIM>,
        sqrt_pi: na::SVectorView<f64, DIM>,
        partial_likelihood: &mut [na::SVector<f64, DIM>],
    ) -> SingleSideResult<f64, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // one out internal nodes in partial_likelihood
        partial_likelihood[self.num_leaves..]
            .iter_mut()
            .for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
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
        // one out internal nodes in partial_likelihood
        partial_likelihood[self.num_leaves..]
            .iter_mut()
            .for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
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
}
