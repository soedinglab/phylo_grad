#![allow(unused, dead_code)]
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

/* Should this get &mut root and modify it in place instead? */
/*
fn log_prob_subtree<const TOTAL: usize>(
    rate_matrix: ArrayView2<Float>,
    node: &BinaryTreeNode<FelsensteinNode>,
) -> Option<Array1<Float>> {
    match (&node.left, &node.right) {
        (Some(left), Some(right)) =>
        // logsumexp_b,c (log_p(left, b) + log_transition(b, a, left.distance) + log_p(right, c) + log_transition(c, a, right.distance))
        // Vectorized: order: b, c, a -> logsumexp( (log_p(left)[:, -1, -1] + log_p(right)[-1, :, -1] + log_transition(left.distance)[:, -1,  :] + log_transition(right.distance)[-1, :, :]).flatten(dim=[0, 1]))
        {
            let mut array3: Array3<Float> = left
                .value
                .log_p
                .to_owned()
                .insert_axis(Axis(1))
                .insert_axis(Axis(2));
            array3 = array3
                + right
                    .value
                    .log_p
                    .view()
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(2));
            array3 = array3 + log_transition(rate_matrix, left.value.distance).insert_axis(Axis(1));
            array3 =
                array3 + log_transition(rate_matrix, right.value.distance).insert_axis(Axis(0));
            Some(logsumexp_a3(array3.view()))
        }
        (Some(left), None) => {
            let mut array2 = left.value.log_p.to_owned().insert_axis(Axis(1));
            array2 = array2 + log_transition(rate_matrix, left.value.distance);
            Some(logsumexp_a2(array2.view()))
        }
        (None, Some(right)) => {
            let mut array2 = right.value.log_p.to_owned().insert_axis(Axis(1));
            array2 = array2 + log_transition(rate_matrix, right.value.distance);
            Some(logsumexp_a2(array2.view()))
        }
        (None, None) => None,
    }
}
*/
pub fn main() {
    let data_path = "data/tree_topological.csv";
    let mut record_reader = read_preprocessed_csv(data_path).unwrap();
    let input: (Vec<TreeNode>, Vec<Option<Vec<ResidueExtended>>>) = record_reader
        .deserialize::<PreprocessedRecord>()
        .map(|x| -> Result<InputTuple, Box<dyn Error>> {
            let y = x?;
            Ok(InputTuple::from(y))
        })
        .map(|x| x.unwrap())
        .unzip();

    let (tree, sequences) = input;

    dbg!(&tree[0..5]);
    dbg!(&sequences[0..5]);

    dbg!(&tree.last());
    dbg!(&sequences.last());

    /*
    let data_path = "data/tree_topological.csv";
    let mut record_reader = read_preprocessed_csv(data_path).unwrap();
    let input = record_reader
        .deserialize::<PreprocessedRecord>()
        .map(|x| -> Result<InputTuple, Box<dyn Error>> {
            let y = x?;
            Ok(InputTuple::from(y))
        })
        .collect::<Result<Vec<InputTuple>, _>>();
    let vector = input.unwrap();
    dbg!(&vector[0..5]);
    dbg!(&vector.last());
    */
}
