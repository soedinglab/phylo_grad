#![allow(non_snake_case)]

extern crate nalgebra as na;


use f64 as Float;
const DIM : usize = 20;

pub fn main() {
    let num_leaf = 100;
    let L = 30;


    let parents = felsenstein_impl::random_tree_top(num_leaf);
    let dist = felsenstein_impl::random_dist::<Float>(parents.len() as u32);

    let leaf_log_p : Vec<Vec<na::SVector<Float, DIM>>> = (0..L).map(|_| {
        (0..num_leaf).map(|_| {
            let init = na::SVector::<Float, DIM>::from_iterator((0..DIM).map(|_| rand::random()));
            let dist = felsenstein_impl::backward::softmax(&init);
            dist.map(Float::ln)
        }).collect()
    }).collect();

    let backend = felsenstein_impl::FelsensteinTree::new(parents, dist, leaf_log_p);
    
    let s : Vec<_> = (0..L).map(|_| {
        na::SMatrix::<Float, DIM, DIM>::from_iterator((0..DIM*DIM).map(|_| rand::random::<Float>().exp()))
    }).collect();

    let sqrt_pi : Vec<_> = (0..L).map(|_| {
        let init = na::SVector::<Float, DIM>::from_iterator((0..DIM).map(|_| rand::random()));
        let dist = felsenstein_impl::backward::softmax(&init);
        dist.map(Float::sqrt)
    }).collect();

    let result = backend.calculate_gradients(s, sqrt_pi);

    std::hint::black_box(result);
}