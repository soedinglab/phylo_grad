#![allow(non_snake_case, clippy::needless_range_loop)]
#![feature(portable_simd)]
extern crate nalgebra as na;

pub mod backward;
pub mod data_types;
mod forward;
mod preprocessing;
mod train;
mod tree;

use rand::distributions::uniform::SampleUniform;
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
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

pub fn random_tree_top(num_leaf: u32) -> Vec<i32> {
    let mut parents = vec![-2;num_leaf as usize];
    let mut orphans : Vec<u32> = (0..num_leaf).collect();

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

pub fn random_dist<F : FloatTrait + SampleUniform>(num_nodes : u32) -> Vec<F> {
    let dist = rand::distributions::Uniform::new::<F, F>(FloatTrait::from_f64(0.1), FloatTrait::from_f64(1.0));
    (0..num_nodes).map(|_| dist.sample(&mut rand::thread_rng())).collect()
}