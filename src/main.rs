#![allow(unused, dead_code)]
//extern crate arrayvec;
extern crate blas_src;
extern crate csv;
extern crate expm;
extern crate logsumexp;
extern crate ndarray;
extern crate num_enum;
extern crate serde;

use expm::expm;
use logsumexp::LogSumExp;
use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::{convert::TryFrom, error::Error, iter::FromIterator};

mod data_types;
mod io;
mod tree;

use crate::data_types::*;
use crate::io::*;
use crate::tree::*;
// use pyo3

/* TODO optimize numeric enum */

/*
fn logsumexp_array<D1: ndarray::Dimension, D2: ndarray::Dimension>(&array: &Array<Float, D1>, axis: Axis) -> Array<Float, D2> {
    /* Expand into an iterator */
    /* TODO rewrite */
    let index = axis.index();
    let mut dim_result = Vec::<usize>::new();
    dim_result.extend(&array.shape()[..index]);
    dim_result.extend(&array.shape()[index+1..]);
    let mut result = unsafe{ ndarray::Array::<Float, _>::uninitialized(dim_result) };
    array.axis_iter(axis).map(LogSumExp::ln_sum_exp).
} */

fn logsumexp_a1(array: ArrayView1<Float>) -> Float {
    array.iter().ln_sum_exp()
}

fn logsumexp_a2(array: ArrayView2<Float>) -> Array1<Float> {
    /* Contracts the first axis */
    /* TODO accept Axis */
    /* expand in an iterator over the remaining axis, apply logsumexp_a1, collect into array */
    array
        .axis_iter(Axis(1))
        .map(logsumexp_a1)
        .collect::<Array1<Float>>()
}

/* TODO SPEED think how this is stored in the memory */
fn logsumexp_a3(array: ArrayView3<Float>) -> Array1<Float> {
    /* Contracts the first two axes */
    array
        .axis_iter(Axis(2))
        .map(|x| x.iter().ln_sum_exp())
        .collect::<Array1<Float>>()
}

/* Should we always get array outputs via a &mut argument? */
fn log_transition(rate_matrix: ArrayView2<Float>, t: Float) -> Array2<Float> {
    let mut result = unsafe { ndarray::Array2::uninitialized(rate_matrix.dim()) };
    /* Is this correct? */
    let mut argument = rate_matrix.to_owned();
    argument *= t;
    expm(&argument, &mut result);
    result.map_inplace(|x| *x = x.ln());
    result
}

/* TODO operate on vectors lazily, perhaps with a buffer, only evaluate when collecting into Array1 at the end. */
/* TODO rewrite to accept individual values */
/* TODO avoid conversion from fixed-size array to ndarray */

fn forward_node(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<LogPType>],
    rate_matrix: ArrayView2<Float>,
) -> Option<Array1<Float>> {
    let node = &tree[id];
    match (node.left, node.right) {
        (Some(left), Some(right)) =>
        // logsumexp_b,c (log_p(left, b) + log_transition(b, a, left.distance) + log_p(right, c) + log_transition(c, a, right.distance))
        // Vectorized: order: b, c, a -> logsumexp( (log_p(left)[:, -1, -1] + log_p(right)[-1, :, -1] + log_transition(left.distance)[:, -1,  :] + log_transition(right.distance)[-1, :, :]).flatten(dim=[0, 1]))
        {
            let log_p_left = log_p[left].unwrap();
            let log_p_right = log_p[right].unwrap();
            let mut array3: Array3<Float> = log_p_left
                .iter()
                .cloned()
                .collect::<Array1<Float>>()
                .insert_axis(Axis(1))
                .insert_axis(Axis(2))
                .broadcast((Entry::DIM, Entry::DIM, Entry::DIM))
                .unwrap()
                .to_owned();
            array3 = array3
                + log_p_right
                    .iter()
                    .cloned()
                    .collect::<Array1<Float>>()
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(2));
            array3 = array3 + log_transition(rate_matrix, tree[left].distance).insert_axis(Axis(1));
            array3 =
                array3 + log_transition(rate_matrix, tree[right].distance).insert_axis(Axis(0));
            Some(logsumexp_a3(array3.view()))
        }
        (Some(left), None) => {
            let log_p_left = log_p[left].unwrap();
            let mut array2 = log_p_left
                .iter()
                .cloned()
                .collect::<Array1<Float>>()
                .insert_axis(Axis(1))
                .broadcast((Entry::DIM, Entry::DIM))
                .unwrap()
                .to_owned();
            array2 = array2 + log_transition(rate_matrix, tree[left].distance);
            Some(logsumexp_a2(array2.view()))
        }
        (None, Some(right)) => {
            let log_p_right = log_p[right].unwrap();
            let mut array2 = log_p_right
                .iter()
                .cloned()
                .collect::<Array1<Float>>()
                .insert_axis(Axis(1))
                .broadcast((Entry::DIM, Entry::DIM))
                .unwrap()
                .to_owned();
            array2 = array2 + log_transition(rate_matrix, tree[right].distance);
            Some(logsumexp_a2(array2.view()))
        }
        (None, None) => None,
    }
}

fn rate_matrix_example() -> Array2<Float> {
    const DIM: usize = Entry::DIM;
    let rate_matrix_example = Array::<Float, _>::eye(DIM)
        - (1 as Float / (DIM - 1) as Float)
            * (Array::<Float, _>::ones((DIM, DIM)) - Array::<Float, _>::eye(DIM));
    rate_matrix_example
}
pub fn main() {
    let data_path = "data/tree_topological.csv";
    let mut record_reader = read_preprocessed_csv(data_path).unwrap();
    /* TODO handle result */
    let tree;
    let mut sequences;
    (tree, sequences) = deserialize_tree(&mut record_reader);

    let num_nodes = tree.len();
    /* TODO terrible */
    let num_leaves = sequences.partition_point(|x| !x.is_empty());
    sequences.truncate(num_leaves);

    let seq_length = sequences[0].len();

    let mut sequences_tmp = Vec::new();

    for i in 0..num_leaves {
        sequences_tmp.extend_from_slice(&sequences[i]);
    }

    let mut sequences_2d = Array2::from_shape_vec((num_leaves, seq_length), sequences_tmp).unwrap();
    /* (!) Does this convert to column major? Seems like it doesn't, as the result has rows with strides=[119] */
    sequences_2d = sequences_2d.t().to_owned();

    let column = sequences_2d.index_axis(Axis(0), 0);
    let mut log_p: Vec<Option<[Float; Entry::DIM]>> =
        column.iter().map(|x| Some(Entry::to_log_p(x))).collect();
    log_p.resize(num_nodes, None);

    let mut rate_matrix = rate_matrix_example();

    let result_nd = forward_node(num_leaves, &tree, &log_p, rate_matrix.view());
    dbg!(result_nd);

    /*
    for i in num_leaves..num_nodes {
        let blah = forward_node(i, &tree, &log_p, rate_matrix.view());
        ...
    }
    */

    /*
    let slice = &column.slice(s![100..120]);
    println!("{:?}", &slice);
    println!("{:?}", &log_p[100..120]);
    */
    //println!("{:?} -> {:?}", &column[0], &log_p[0]);
}
