#![allow(non_snake_case)]
#![feature(portable_simd)]

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

pub use data_types::FloatTrait;

mod backward;
mod data_types;
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
///
/// It is also generic over `f32` and `f64`
pub struct FelsensteinTree<F, const DIM: usize> {
    parents: Vec<i32>,
    distances: Vec<F>,
    num_leaves: usize,
    log_p: Vec<Vec<na::SVector<F, DIM>>>,
    tmp_mem: Option<Vec<Vec<na::SMatrix<F, DIM, DIM>>>>,
    sorting: Vec<usize>,
}

impl<F: FloatTrait, const DIM: usize> FelsensteinTree<F, DIM> {
    /// The tree topology is represented as a vector of parent node ids. The root node has parent id `-1`.
    /// The leaf nodes have to come first in this slice.
    ///
    /// The distances are given as a vector of branch lengths with the same order as the parent vector.
    pub fn new(parents: &[i32], distances: &[F]) -> Self {
        assert!(parents.len() == distances.len());
        let (parents, distances, num_leaves, sorting) = tree::topological_sort(parents, distances);

        assert_eq!(parents.last().unwrap(), &-1); // root node is the last node

        FelsensteinTree {
            parents,
            distances,
            log_p: vec![],
            num_leaves,
            tmp_mem: None,
            sorting,
        }
    }

    /// Binds the log probabilities of the leaves to the tree.
    /// This enables usage of the `calculate_gradients` function.
    pub fn bind_leaf_log_p(&mut self, log_p: Vec<Vec<na::SVector<F, DIM>>>) {
        self.log_p = log_p;
        // resize the log_p to the number of all nodes
        let num_nodes = self.parents.len();
        for log_p in &mut self.log_p {
            log_p.resize(num_nodes, na::SVector::<F, DIM>::zeros());
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.parents.len()
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
    /// This functions assumes you have already called `bind_leaf_log_p` to bind the log probabilities of the leaves.
    pub fn calculate_gradients(
        &mut self,
        s: &[na::SMatrix<F, DIM, DIM>],
        sqrt_pi: &[na::SVector<F, DIM>],
    ) -> FelsensteinResult<F, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // Zero out internal nodes in log_p
        for log_p in &mut self.log_p {
            log_p.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<F, DIM>::zeros();
            });
        }

        if s.len() == 1 && sqrt_pi.len() == 1 {
            let d_trans_matrix = self.tmp_mem.get_or_insert_with(|| {
                let num_nodes = self.parents.len();
                let L = self.log_p.len();
                vec![vec![na::SMatrix::<F, DIM, DIM>::zeros(); num_nodes]; L]
            });

            return calculate_column_parallel_single_S(
                &mut self.log_p,
                &s[0],
                &sqrt_pi[0],
                tree,
                d_trans_matrix,
                false,
                None,
            );
        }
        calculate_column_parallel(&mut self.log_p, s, sqrt_pi, tree, false)
    }

    pub fn calculate_edge_gradients(
        &mut self,
        s: &na::SMatrix<F, DIM, DIM>,
        sqrt_pi: &na::SVector<F, DIM>,
    ) -> Vec<F> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // Zero out internal nodes in log_p
        for log_p in &mut self.log_p {
            log_p.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<F, DIM>::zeros();
            });
        }

        let d_trans_matrix = self.tmp_mem.get_or_insert_with(|| {
                let num_nodes = self.parents.len();
                let L = self.log_p.len();
                vec![vec![na::SMatrix::<F, DIM, DIM>::zeros(); num_nodes]; L]
        });

        let mut grad_edges = vec![F::zero(); self.distances.len()];

        calculate_column_parallel_single_S(
            &mut self.log_p,
            s,
            sqrt_pi,
            tree,
            d_trans_matrix,
            false,
            Some(&mut grad_edges)
        );

        // reorder gradients to original order
        let grad_edges_reordered = {
            let mut g = vec![F::zero(); grad_edges.len()];
            for (new_idx, &old_idx) in self.sorting.iter().enumerate() {
                g[old_idx] = grad_edges[new_idx];
            }
            g
        };

        grad_edges_reordered
    }

    /// Same as `calculate_gradients`, but only calculates the log likelihoods for each side in the alignment.
    pub fn calculate_likelihoods(
        &mut self,
        s: &[na::SMatrix<F, DIM, DIM>],
        sqrt_pi: &[na::SVector<F, DIM>],
    ) -> Vec<F> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // Zero out internal nodes in log_p
        for log_p in &mut self.log_p {
            log_p.iter_mut().skip(self.num_leaves).for_each(|p| {
                *p = na::SVector::<F, DIM>::zeros();
            });
        }

        let result = if s.len() == 1 && sqrt_pi.len() == 1 {
            let d_trans_matrix = self.tmp_mem.get_or_insert_with(|| {
                let num_nodes = self.parents.len();
                let L = self.log_p.len();
                vec![vec![na::SMatrix::<F, DIM, DIM>::zeros(); num_nodes]; L]
            });

            calculate_column_parallel_single_S(
                &mut self.log_p,
                &s[0],
                &sqrt_pi[0],
                tree,
                d_trans_matrix,
                true,
                None
            )
        } else {
            calculate_column_parallel(&mut self.log_p, s, sqrt_pi, tree, true)
        };

        return result.log_likelihood;
    }

    /// Same as `calculate_gradients`, but it takes also an array of the log_probabilities of the leaves.
    /// It expects `log_p` to have enough space for all nodes with internal nodes initialized to zero and leaf nodes properly initialized.
    pub fn calculate_gradients_with_log_p(
        &self,
        s: &[na::SMatrix<F, DIM, DIM>],
        sqrt_pi: &[na::SVector<F, DIM>],
        log_p: &mut [&mut [na::SVector<F, DIM>]],
    ) -> FelsensteinResult<F, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        calculate_column_parallel(log_p, s, sqrt_pi, tree, false)
    }

    /// This function calculates the gradients for a single side in the alignment.
    /// This can be useful if you want to control the parallelization yourself or if you want to calculate the gradients for a single side.
    ///
    /// log_p is expected to have enough space to hold the log probabilities for all nodes
    pub fn calculate_gradients_single_side(
        &self,
        s: na::SMatrixView<F, DIM, DIM>,
        sqrt_pi: na::SVectorView<F, DIM>,
        log_p: &mut [na::SVector<F, DIM>],
    ) -> SingleSideResult<F, DIM> {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // zero out internal nodes in log_p
        log_p[self.num_leaves..].iter_mut().for_each(|p| {
            *p = na::SVector::<F, DIM>::zeros();
        });
        calculate_column(log_p, s.as_view(), sqrt_pi.as_view(), tree, false)
    }

    pub fn calculate_likelihood_single_side(
        &self,
        s: na::SMatrixView<F, DIM, DIM>,
        sqrt_pi: na::SVectorView<F, DIM>,
        log_p: &mut [na::SVector<F, DIM>],
    ) -> F {
        let tree = tree::Tree::new(&self.parents, &self.distances, self.num_leaves);
        // zero out internal nodes in log_p
        log_p[self.num_leaves..].iter_mut().for_each(|p| {
            *p = na::SVector::<F, DIM>::zeros();
        });
        let result = calculate_column(log_p, s.as_view(), sqrt_pi.as_view(), tree, true);
        result.log_likelihood
    }
}
