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
mod numerics;
mod forward;
mod run;
mod tree;

pub use run::FelsensteinResult;
pub use run::SingleSideResult;

use crate::run::*;

/// Represents a tree topology with branch lengths
/// This struct contains the main functionality of the library.
///
/// It is generic over the number of states in the model, which is given by `DIM`.
pub struct FelsensteinTree<const DIM: usize> {
    parents: Vec<i32>,
    distances: Vec<f64>,
    num_leaves: usize,
    /// First dimension is the side id in the alignment, second dimension is the node id in the tree.
    partial_likelihoods: Vec<Vec<na::SVector<f64, DIM>>>,
}

impl<const DIM: usize> FelsensteinTree<DIM> {
    /// The tree topology is represented as a vector of parent node ids. The root node has parent id `-1`.
    /// The leaf nodes have to come first in this slice.
    ///
    /// The distances are given as a vector of branch lengths with the same order as the parent vector.
    pub fn new(parents: &[i32], distances: &[f64]) -> Self {
        assert!(parents.len() == distances.len());
        let (parents, distances, num_leaves) = tree::topological_sort(parents, distances);

        assert_eq!(parents.last().unwrap(), &-1); // root node is the last node

        FelsensteinTree {
            parents,
            distances,
            partial_likelihoods: vec![],
            num_leaves,
        }
    }

    /// Binds the log probabilities of the leaves to the tree.
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
    ) -> FelsensteinResult<f64, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // One out internal nodes in partial_likelihoods
        for pl in &mut self.partial_likelihoods {
            pl.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<f64, DIM>::from_element(1.0);
            });
        }

        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            todo!();
        } else {
            calculate_column_parallel(&mut self.partial_likelihoods, s, sqrt_pi, tree, false)
        };
        result
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
            todo!();
        } else {
            calculate_column_parallel(&mut self.partial_likelihoods, s, sqrt_pi, tree, true)
        };

        return result.log_likelihood;
    }

    /// Same as `calculate_gradients`, but it takes also an array of the partial likelihoods of the leaves.
    /// It expects `pl` to have enough space for all nodes with internal nodes initialized to one and leaf nodes properly initialized.
    pub fn calculate_gradients_with_pl_vec(
        &self,
        s: &[na::SMatrix<f64, DIM, DIM>],
        sqrt_pi: &[na::SVector<f64, DIM>],
        partial_likelihood: &mut [&mut [na::SVector<f64, DIM>]],
    ) -> FelsensteinResult<f64, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        calculate_column_parallel(partial_likelihood, s, sqrt_pi, tree, false)
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
        partial_likelihood[self.num_leaves..].iter_mut().for_each(|p| {
            *p = na::SVector::<f64, DIM>::from_element(1.0);
        });
        calculate_column(partial_likelihood, s.as_view(), sqrt_pi.as_view(), tree, false)
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
        partial_likelihood[self.num_leaves..].iter_mut().for_each(|p| {
            *p = na::SVector::<f64, DIM>::from_element(1.0);
        });
        let result = calculate_column(partial_likelihood, s.as_view(), sqrt_pi.as_view(), tree, true);
        result.log_likelihood
    }
}
