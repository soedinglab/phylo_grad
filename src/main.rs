#![allow(unused, dead_code)]
//extern crate arrayvec;
//extern crate blas_src;
extern crate csv;
extern crate logsumexp;
extern crate nalgebra as na;
extern crate ndarray;
extern crate num_enum;
extern crate serde;
/* TODO pyo3 */

use logsumexp::LogSumExp;
use ndarray::{Array2, Axis};
use std::error::Error;

mod backward;
mod data_types;
mod forward;
mod io;
mod tree;

use crate::backward::*;
use crate::data_types::*;
use crate::forward::*;
use crate::io::*;
use crate::tree::*;

fn rate_matrix_example() -> RateType {
    const DIM: usize = Entry::DIM;
    let rate_matrix_example = -RateType::identity()
        + (1 as Float / (DIM - 1) as Float)
            * (RateType::from_element(1.0 as Float) - RateType::identity());
    rate_matrix_example
}

pub fn main() {
    /* TODO get rid of Options */
    /* TODO! apply cutoff for distances */
    /* Placeholder values */
    let args: Vec<String> = std::env::args().collect();

    let log_prior = [(Entry::DIM as Float).recip(); Entry::DIM].map(Float::ln);
    let rate_matrix = rate_matrix_example();

    let data_path = if args.len() >= 2 {
        &args[1]
    } else {
        "data/tree_topological.csv"
    };
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

    let mut log_likelihood = 0.0 as Float;
    let mut grad_log_prior = [0.0 as Float; Entry::DIM];
    for (column_id, column) in sequences_2d.axis_iter(Axis(0)).enumerate() {
        let mut log_p: Vec<Option<[Float; Entry::DIM]>> =
            column.iter().map(|x| Some(Entry::to_log_p(x))).collect();
        log_p.resize(num_nodes, None);

        for i in num_leaves..(num_nodes - 1) {
            let log_p_new = forward_node(i, &tree, &log_p, rate_matrix.as_view()).unwrap();
            log_p[i] = Some(log_p_new);
        }
        let log_p_root = forward_root(num_nodes - 1, &tree, &log_p, rate_matrix.as_view());
        log_p[num_leaves - 1] = Some(log_p_root);

        /* TODO macro */
        let mut _BACKWARD_likelihood_lse_arg = [0.0 as Float; Entry::DIM];
        for i in (0..Entry::DIM) {
            _BACKWARD_likelihood_lse_arg[i] = log_p_root[i] + log_prior[i];
        }
        let log_likelihood_column = _BACKWARD_likelihood_lse_arg.iter().ln_sum_exp();

        softmax_inplace(&mut _BACKWARD_likelihood_lse_arg);
        for i in (0..Entry::DIM) {
            grad_log_prior[i] += _BACKWARD_likelihood_lse_arg[i];
        }

        /*
        let log_likelihood_column = (0..Entry::DIM)
            .map(|i| log_p_root[i] + log_prior[i])
            .ln_sum_exp();
        */
        log_likelihood += log_likelihood_column;
        println!("Log likelihood #{} = {}", column_id, log_likelihood_column);
    }
    println!("Log likelihood = {}", log_likelihood);
    println!("Gradient of log_prior = {:?}", grad_log_prior);
}
