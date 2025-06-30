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

pub use backward::softmax;
pub use data_types::FloatTrait;

mod backward;
mod data_types;
mod forward;
mod preprocessing;
mod run;
mod tree;

use rand::distributions::uniform::SampleUniform;
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
pub use run::FelsensteinResult;

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
}

/// Returns a random tree topology in the format required by FelsensteinTree::new
pub fn random_tree_top(num_leaf: u32) -> Vec<i32> {
    let mut parents = vec![-2; num_leaf as usize];
    let mut orphans: Vec<u32> = (0..num_leaf).collect();

    let mut rng = rand::thread_rng();

    while orphans.len() > 3 {
        // This is inefficient, but it's not a critical part of the pipeline
        orphans.shuffle(&mut rng);
        let parent_id = parents.len();
        parents.push(-2);
        let sib1 = orphans.pop().unwrap();
        let sib2 = orphans.pop().unwrap();

        parents[sib1 as usize] = parent_id as i32;
        parents[sib2 as usize] = parent_id as i32;

        orphans.push(parent_id as u32);
    }

    let root_id = parents.len();
    let sib1 = orphans.pop().unwrap();
    let sib2 = orphans.pop().unwrap();
    let sib3 = orphans.pop().unwrap();
    parents[sib1 as usize] = root_id as i32;
    parents[sib2 as usize] = root_id as i32;
    parents[sib3 as usize] = root_id as i32;

    parents.push(-1);

    parents
}

/// Returns a random distance vector in the format required by FelsensteinTree::new uniformly distributed between 0.1 and 1.0
pub fn random_dist<F: FloatTrait + SampleUniform>(num_nodes: u32) -> Vec<F> {
    let dist = rand::distributions::Uniform::new::<F, F>(
        FloatTrait::from_f64(0.1),
        FloatTrait::from_f64(1.0),
    );
    (0..num_nodes)
        .map(|_| dist.sample(&mut rand::thread_rng()))
        .collect()
}
