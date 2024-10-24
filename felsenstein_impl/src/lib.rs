#![allow(non_snake_case, clippy::needless_range_loop)]
extern crate nalgebra as na;

mod backward;
pub mod data_types;
mod forward;
mod preprocessing;
mod train;
mod tree;

pub use train::InferenceResultParam;

use crate::data_types::*;
use crate::preprocessing::*;
use crate::train::*;
use crate::tree::*;

pub struct FTreeBackend<F, const DIM: usize> {
    tree: Vec<TreeNodeId<u32>>,
    distances: Vec<F>,
    leaf_log_p: Vec<Vec<na::SVector<F, DIM>>>,
}

impl<F: FloatTrait, const DIM : usize> FTreeBackend<F, DIM> {
    pub fn new(parents : Vec<i32>, distances : Vec<F>, leaf_log_p : Vec<Vec<na::SVector<F, DIM>>>) -> Self {
        let (tree, distances) = topological_preprocess::<F>(parents, distances, leaf_log_p[0].len() as u32).expect("Tree topology is invalid");

        FTreeBackend {
            tree,
            distances,
            leaf_log_p,
        }
    }

    pub fn infer(&self, s: Vec<na::SMatrix<F, DIM, DIM>>, sqrt_pi: Vec<na::SVector<F, DIM>>) -> InferenceResultParam<F, DIM> {
        train_parallel_param_unpaired(&self.leaf_log_p, &s, &sqrt_pi, &self.tree, &self.distances)
    }
}