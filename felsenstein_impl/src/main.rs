#![allow(non_snake_case)]

use rand::{distributions::uniform::SampleUniform, prelude::Distribution, seq::SliceRandom};
extern crate nalgebra as na;

use felsenstein_impl::data_types::*;

pub fn gen_tree_top(num_leaf: u32) -> Vec<i32> {
    let mut parents = vec![-2;num_leaf as usize];
    let mut orphans : Vec<u32> = (0..num_leaf).collect();

    let mut rng = rand::thread_rng();

    while orphans.len() > 3 {
        // This is inefficient, but it's not a bottleneck
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

pub fn gen_dist<F : FloatTrait + SampleUniform>(num_nodes : u32) -> Vec<F> {
    let dist = rand::distributions::Uniform::new::<F, F>(FloatTrait::from_f64(0.1), FloatTrait::from_f64(1.0));
    (0..num_nodes).map(|_| dist.sample(&mut rand::thread_rng())).collect()
}

use f64 as Float;
const DIM : usize = 20;

pub fn main() {
    let num_leaf = 10;
    let L = 300;


    let parents = gen_tree_top(num_leaf);
    let dist = gen_dist::<Float>(parents.len() as u32);

    let leaf_log_p : Vec<Vec<na::SVector<Float, DIM>>> = (0..L).map(|_| {
        (0..num_leaf).map(|_| {
            let init = na::SVector::<Float, DIM>::from_iterator((0..DIM).map(|_| rand::random()));
            let dist = felsenstein_impl::backward::softmax(init.as_view());
            dist.map(Float::ln)
        }).collect()
    }).collect();

    let backend = felsenstein_impl::FTreeBackend::new(parents, dist, leaf_log_p);
    
    let s : Vec<_> = (0..L).map(|_| {
        na::SMatrix::<Float, DIM, DIM>::from_iterator((0..DIM*DIM).map(|_| rand::random::<Float>().exp()))
    }).collect();

    let sqrt_pi : Vec<_> = (0..L).map(|_| {
        let init = na::SVector::<Float, DIM>::from_iterator((0..DIM).map(|_| rand::random()));
        let dist = felsenstein_impl::backward::softmax(init.as_view());
        dist.map(Float::sqrt)
    }).collect();

    let result = backend.infer(s, sqrt_pi);

    std::hint::black_box(result);
}