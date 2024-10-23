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

pub fn main() {
    let parents = gen_tree_top(10);
    let dist = gen_dist::<Float>(parents.len() as u32);



}