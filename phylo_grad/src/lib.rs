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

/// Used to abstract over `f32` and `f64` in the crate. It is not supposed to be used outside of this crate except for trait bounds.
pub use data_types::FloatTrait;

mod backward;
mod data_types;
mod forward;
mod preprocessing;
mod run;
mod tree;

pub use run::FelsensteinResult;
pub use run::SingleSideResult;

use crate::preprocessing::*;
use crate::run::*;
use crate::tree::*;

/// Represents a tree topology, branch length and leaf node data
/// This struct contains the main functionality of the library.
///
/// It is generic over the number of states in the model, which is given by `DIM`.
///
/// It is generic over `f32` and `f64`
pub struct FelsensteinTree<F, const DIM: usize> {
    tree: Vec<TreeNodeId<u32>>,
    distances: Vec<F>,
    leaf_log_p: Vec<Vec<na::SVector<F, DIM>>>,
    tmp_mem: Option<Vec<Vec<na::SMatrix<F, DIM, DIM>>>>,
}

impl<F: FloatTrait, const DIM: usize> FelsensteinTree<F, DIM> {
    /// The tree topology is represented as a vector of parent node ids. The root node has parent id `-1`.
    /// The leaf nodes have to come first in this vector. The order of the leaf nodes is the same as the order of the second dimension of `leaf_log_p`.
    ///
    /// The distances are given as a vector of branch lengths with the same order as the parent vector.
    ///
    /// `leaf_log_p` gives the log probabilities of the leaf nodes. The first dimension is the side id in the alignment, the second is over the leaf nodes, the third is over the states.
    pub fn new(
        parents: Vec<i32>,
        distances: Vec<F>,
        leaf_log_p: Vec<Vec<na::SVector<F, DIM>>>,
    ) -> Self {
        let (tree, distances) =
            topological_preprocess::<F>(parents, distances, leaf_log_p[0].len() as u32)
                .expect("Tree topology is invalid");

        FelsensteinTree {
            tree,
            distances,
            leaf_log_p,
            tmp_mem: None,
        }
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
    pub fn calculate_gradients(
        &mut self,
        s: Vec<na::SMatrix<F, DIM, DIM>>,
        sqrt_pi: Vec<na::SVector<F, DIM>>,
    ) -> FelsensteinResult<F, DIM> {
        if s.len() == 1 && sqrt_pi.len() == 1 {
            let d_trans_matrix = self.tmp_mem.get_or_insert_with(|| {
                let num_nodes = self.tree.len();
                let L = self.leaf_log_p.len();
                vec![vec![na::SMatrix::<F, DIM, DIM>::zeros(); num_nodes]; L]
            });

            return calculate_column_parallel_single_S(
                &self.leaf_log_p,
                &s[0],
                &sqrt_pi[0],
                &self.tree,
                &self.distances,
                d_trans_matrix,
            );
        }
        calculate_column_parallel(&self.leaf_log_p, &s, &sqrt_pi, &self.tree, &self.distances)
    }

    /// This function calculates the gradients for a single side in the alignment.
    /// This can be useful if you want to control the parallelization yourself or if you want to calculate the gradients for a single side.
    pub fn calculate_gradients_single_side(
        &self,
        s: na::SMatrix<F, DIM, DIM>,
        sqrt_pi: na::SVector<F, DIM>,
        side_id: usize,
    ) -> SingleSideResult<F, DIM> {
        let leaf_log_p = self.leaf_log_p[side_id].clone();

        calculate_column(leaf_log_p, s.as_view(), sqrt_pi.as_view(), &self.tree, &self.distances)
    }
}