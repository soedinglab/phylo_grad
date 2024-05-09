#![allow(unused, dead_code)]
//extern crate arrayvec;
//extern crate blas_src;
extern crate csv;
extern crate itertools;
extern crate logsumexp;
extern crate nalgebra as na;
extern crate ndarray;
extern crate num_enum;
extern crate serde;
/* TODO pyo3 */

use itertools::process_results;
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

fn print_vector<T>(vector: na::DVectorView<T>)
where
    T: std::fmt::Debug + Clone,
{
    dbg!(
        &vector
            .column(0)
            .iter()
            .map(|x| (*x).clone())
            .collect::<Vec<T>>()[0..20]
    );
}

fn rate_matrix_example() -> RateType {
    const DIM: usize = Entry::DIM;
    let rate_matrix_example = -RateType::identity()
        + (1 as Float / (DIM - 1) as Float)
            * (RateType::from_element(1.0 as Float) - RateType::identity());
    rate_matrix_example
}
/* TODO don't return a tuple, even &mut result is better */
fn try_entry_sequences_from_strings(
    sequences_raw: &[Option<String>],
) -> (usize, usize, na::DMatrix<Entry>) {
    /* fn(sequences_raw) -> (sequences_2d, num_leaves, seq_length) */
    let num_leaves = sequences_raw.partition_point(|x| !x.is_none());

    let seq_expected_length = sequences_raw[0]
        .as_ref()
        .expect("The first node has to be a leaf")
        .len()
        / Entry::CHARS;

    let mut sequences_flat = Vec::<Entry>::with_capacity(num_leaves * seq_expected_length);

    sequences_raw[0..num_leaves]
        .into_iter()
        .map(|x| x.as_ref().unwrap())
        .map(|s: &String| {
            process_results(Entry::try_deserialize_string_iter(s), |it| {
                sequences_flat.extend(it)
            })
        })
        .for_each(|x| x.unwrap());
    /* TODO get actual length from iterator */
    let seq_length = sequences_flat.len() / num_leaves;

    (
        num_leaves,
        seq_length,
        na::DMatrix::from_vec_generic(na::Dyn(num_leaves), na::Dyn(seq_length), sequences_flat),
    )
}

/* TODO get rid of ndarray */
fn try_entry_sequences_from_strings_ndarray(
    sequences_raw: &mut Vec<Option<String>>,
) -> (usize, usize, Array2<Entry>) {
    let num_leaves = sequences_raw.partition_point(|x| !x.is_none());
    sequences_raw.truncate(num_leaves);

    let sequences: Vec<Vec<Entry>> = sequences_raw
        .iter()
        .map(|x| x.as_ref().unwrap())
        .map(|x: &String| {
            process_results(Entry::try_deserialize_string_iter(x), |it| -> Vec<Entry> {
                it.collect()
            })
        })
        .map(|x| x.unwrap())
        .collect();
    //let seq_length_raw = sequences_raw[0].as_ref().unwrap().len();
    //let seq_length_raw_adjusted = seq_length_raw - (seq_length_raw % Entry::CHARS);
    let seq_length = sequences[0].len();

    let mut sequences_tmp = Vec::with_capacity(num_leaves * seq_length);

    for i in 0..num_leaves {
        sequences_tmp.extend_from_slice(&sequences[i]);
    }

    let mut sequences_2d = Array2::from_shape_vec((num_leaves, seq_length), sequences_tmp).unwrap();
    /* TODO this does not transpose sequences_2d in the memory, fix this */
    sequences_2d = sequences_2d.t().to_owned();
    /* --- Finished transposing the sequences : ndarray --- *//* --- Transposing the sequences: ndarray --- */
    /* TODO get rid of ndarray */
    let num_leaves = sequences_raw.partition_point(|x| !x.is_none());
    sequences_raw.truncate(num_leaves);

    let sequences: Vec<Vec<Entry>> = sequences_raw
        .into_iter()
        .map(|x| x.as_ref().unwrap())
        .map(|x: &String| {
            process_results(Entry::try_deserialize_string_iter(x), |it| -> Vec<Entry> {
                it.collect()
            })
        })
        .map(|x| x.unwrap())
        .collect();
    //let seq_length_raw = sequences_raw[0].as_ref().unwrap().len();
    //let seq_length_raw_adjusted = seq_length_raw - (seq_length_raw % Entry::CHARS);
    let seq_length = sequences[0].len();

    let mut sequences_tmp = Vec::with_capacity(num_leaves * seq_length);

    for i in 0..num_leaves {
        sequences_tmp.extend_from_slice(&sequences[i]);
    }

    let mut sequences_2d = Array2::from_shape_vec((num_leaves, seq_length), sequences_tmp).unwrap();
    /* TODO this does not transpose sequences_2d in the memory, fix this */
    sequences_2d = sequences_2d.t().to_owned();
    (num_leaves, seq_length, sequences_2d)
}

fn forward_column(
    column: na::DVectorView<Entry>,
    tree: &[TreeNode],
    distances: &[Float],
    log_p: &mut Vec<Option<[Float; Entry::DIM]>>,
    forward_data: &mut Vec<LogTransitionForwardData<{ Entry::DIM }>>,
    rate_matrix: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    num_nodes: Id,
    num_leaves: Id,
) {
    forward_data_precompute(forward_data, rate_matrix, distances, (0..num_nodes));

    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    log_p.clear();
    log_p.extend(column.iter().map(|x| Some(Entry::to_log_p(x))));
    log_p.resize(num_nodes, None);

    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, &tree, &log_p, &forward_data).unwrap();
        log_p[i] = Some(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, &tree, &log_p, &forward_data);
    log_p[num_leaves - 1] = Some(log_p_root);
}

fn forward_column_ndarray(
    column: ndarray::ArrayView1<Entry>,
    tree: &[TreeNode],
    distances: &[Float],
    log_p: &mut Vec<Option<[Float; Entry::DIM]>>,
    forward_data: &mut Vec<LogTransitionForwardData<{ Entry::DIM }>>,
    rate_matrix: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
    num_nodes: Id,
    num_leaves: Id,
) {
    forward_data_precompute(forward_data, rate_matrix, distances, (0..num_nodes));

    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    log_p.clear();
    log_p.extend(column.iter().map(|x| Some(Entry::to_log_p(x))));
    log_p.resize(num_nodes, None);

    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, &tree, &log_p, &forward_data).unwrap();
        log_p[i] = Some(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, &tree, &log_p, &forward_data);
    log_p[num_leaves - 1] = Some(log_p_root);
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    /* Placeholder values */
    let log_prior = [(Entry::DIM as Float).recip(); Entry::DIM].map(Float::ln);
    let rate_matrix = rate_matrix_example();
    let distance_threshold = 1e-9 as Float;

    let data_path = if args.len() >= 2 {
        &args[1]
    } else {
        "data/tree_topological.csv"
    };

    let mut record_reader = read_preprocessed_csv(data_path).unwrap();
    /* TODO handle result */
    let tree;
    let mut distances;
    let mut sequences_raw;
    (tree, distances, sequences_raw) = deserialize_tree(&mut record_reader).unwrap();

    distances
        .iter_mut()
        .for_each(|d| *d = distance_threshold.max(*d));

    let num_nodes = tree.len();

    let mut forward_data =
        Vec::<LogTransitionForwardData<{ Entry::DIM }>>::with_capacity(num_nodes);
    /* TODO get rid of Options */
    let mut log_p = Vec::<Option<[Float; Entry::DIM]>>::with_capacity(num_nodes);

    let (num_leaves_na, seq_length_na, sequences_2d_na) =
        try_entry_sequences_from_strings(&sequences_raw);

    let (num_leaves, seq_length, sequences_2d) =
        try_entry_sequences_from_strings_ndarray(&mut sequences_raw);

    //return;
    let mut log_likelihood = 0.0 as Float;
    let mut grad_log_prior = [0.0 as Float; Entry::DIM];

    //for (column_id, column) in sequences_2d.column_iter().enumerate() {
    for (column_id, column) in sequences_2d.axis_iter(Axis(0)).enumerate() {
        /* Right now, this is the same for all columns, but as every column will have its own
        rate matrix, in general we'll have to precompute log_transition for each column*/

        forward_column_ndarray(
            column.view(),
            &tree,
            &distances,
            &mut log_p,
            &mut forward_data,
            rate_matrix.as_view(),
            num_nodes,
            num_leaves,
        );

        let log_p_root = log_p[num_leaves - 1].unwrap();

        let mut forward_data_likelihood_lse_arg = [0.0 as Float; Entry::DIM];
        for i in (0..Entry::DIM) {
            forward_data_likelihood_lse_arg[i] = log_p_root[i] + log_prior[i];
        }
        let log_likelihood_column = forward_data_likelihood_lse_arg.iter().ln_sum_exp();

        softmax_inplace(&mut forward_data_likelihood_lse_arg);
        for i in (0..Entry::DIM) {
            grad_log_prior[i] += forward_data_likelihood_lse_arg[i];
        }

        log_likelihood += log_likelihood_column;
        println!("Log likelihood #{} = {}", column_id, log_likelihood_column);
    }
    println!("Log likelihood = {}", log_likelihood);
    println!("Gradient of log_prior = {:?}", grad_log_prior);
}
