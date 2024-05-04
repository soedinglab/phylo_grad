#![allow(unused, dead_code)]
//extern crate arrayvec;
//extern crate blas_src;
extern crate csv;
extern crate logsumexp;
extern crate nalgebra as na;
extern crate ndarray;

extern crate num_enum;
extern crate serde;

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

fn log_transition<const DIM: usize>(
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
    t: Float,
) -> na::SMatrix<Float, DIM, DIM>
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let argument = rate_matrix.map(Float::exp) * t;
    let matrix_exp = argument.exp();
    matrix_exp.map(Float::ln)
}

fn _child_input<const DIM: usize>(
    log_p: &[Float; DIM],
    distance: Float,
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM]
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    /* result_a = logsumexp_b(log_p(b) + log_transition(rate_matrix, distance)(b, a) ) */
    let log_transition = log_transition(rate_matrix, distance);
    /* Is this better or worse than adding two nalgebra vectors and taking logsumexp? */
    let mut result = [0.0 as Float; DIM];
    for a in (0..DIM) {
        result[a] = (0..DIM)
            .map(|b| (log_p[b] + log_transition[(a, b)]))
            .ln_sum_exp()
    }
    result
}

/* TODO duplicate code */
fn forward_node<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<[Float; DIM]>],
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> Option<[Float; DIM]>
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let node = &tree[id];
    match (node.left, node.right) {
        (Some(left), Some(right)) => {
            let log_p_left = log_p[left].unwrap();
            let child_input_left = _child_input(&log_p_left, tree[left].distance, rate_matrix);

            let log_p_right = log_p[right].unwrap();
            let child_input_right = _child_input(&log_p_right, tree[right].distance, rate_matrix);

            let mut result = [0.0 as Float; DIM];
            for a in (0..DIM) {
                result[a] = child_input_left[a] + child_input_right[a];
            }
            Some(result)
        }
        (Some(left), None) => {
            let log_p_left = log_p[left].unwrap();
            let result = _child_input(&log_p_left, tree[left].distance, rate_matrix);
            Some(result)
        }
        (None, Some(right)) => {
            let log_p_right = log_p[right].unwrap();
            let result = _child_input(&log_p_right, tree[right].distance, rate_matrix);
            Some(result)
        }
        (None, None) => None,
    }
}

fn forward_root<const DIM: usize>(
    id: Id,
    tree: &[TreeNode],
    log_p: &[Option<[Float; DIM]>],
    rate_matrix: na::SMatrixView<Float, DIM, DIM>,
) -> [Float; DIM]
where
    na::Const<DIM>: na::ToTypenum,
    na::Const<DIM>: na::DimMin<na::Const<DIM>, Output = na::Const<DIM>>,
{
    let root = &tree[id];

    let mut children = Vec::with_capacity(3);
    children.push(root.parent);
    if let Some(child) = root.left {
        children.push(child);
    }
    if let Some(child) = root.right {
        children.push(child);
    }

    let result: na::SVector<Float, DIM> = children
        .into_iter()
        .map(|child| {
            let log_p_child = log_p[child].unwrap();
            na::SVector::<Float, DIM>::from(_child_input(
                &log_p_child,
                tree[child].distance,
                rate_matrix,
            ))
        })
        .sum();
    <[Float; DIM]>::from(result)
}

fn rate_matrix_example() -> RateType {
    const DIM: usize = Entry::DIM;
    let rate_matrix_example = RateType::identity()
        - (1 as Float / (DIM - 1) as Float)
            * (RateType::from_element(1.0 as Float) - RateType::identity());
    rate_matrix_example
}

pub fn main() {
    /* Placeholder values */
    let log_prior = [0.25 as Float; Entry::DIM].map(Float::ln);
    let rate_matrix = rate_matrix_example();

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

    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, &tree, &log_p, rate_matrix.as_view()).unwrap();
        log_p[i] = Some(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, &tree, &log_p, rate_matrix.as_view());

    let log_likelihood = (0..Entry::DIM)
        .map(|i| log_p_root[i] + log_prior[i])
        .ln_sum_exp();

    println!("Log likelihood = {}", log_likelihood);
}
