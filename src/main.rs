#![allow(unused, dead_code)]
//extern crate arrayvec;
//extern crate blas_src;
extern crate csv;
extern crate itertools;
extern crate logsumexp;
extern crate nalgebra as na;
extern crate num_enum;
extern crate serde;
/* TODO pyo3 */

use itertools::process_results;
use logsumexp::LogSumExp;
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

impl FelsensteinError {
    pub const ORDER: Self = Self::DeserializationError("The tree is not topoligically ordered");
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
) -> Result<(usize, usize, na::DMatrix<Entry>), FelsensteinError> {
    let num_leaves = sequences_raw.partition_point(|x| !x.is_none());

    let seq_expected_length = sequences_raw[0]
        .as_ref()
        .expect("The first node has to be a leaf")
        .len()
        / Entry::CHARS;

    let mut sequences_flat = Vec::<Entry>::with_capacity(num_leaves * seq_expected_length);

    for res in sequences_raw[0..num_leaves]
        .into_iter()
        .map(|x| match x.as_ref() {
            Some(s) => Ok(s),
            None => Err(FelsensteinError::ORDER),
        })
        .map(|res: Result<&String, FelsensteinError>| {
            process_results(Entry::try_deserialize_string_iter(res?), |it| {
                sequences_flat.extend(it)
            })
        })
    {
        if let Err(err) = res {
            return Err(err);
        }
    }
    /* TODO get actual length from iterator, check all lengths are the same */
    let seq_length = sequences_flat.len() / num_leaves;
    let sequences_2d =
        na::DMatrix::from_vec_generic(na::Dyn(seq_length), na::Dyn(num_leaves), sequences_flat);
    Ok((num_leaves, seq_length, sequences_2d))
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
    /* Right now, this is the same for all columns, but as every column will have its own
    rate matrix, in general we'll have to precompute log_transition for each column */
    forward_data_precompute(forward_data, rate_matrix, distances, (0..num_nodes));

    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    log_p.clear();
    log_p.extend(column.iter().map(|x| Some(Entry::to_log_p(x))));
    log_p.resize(num_nodes, None);

    /* TODO remove copy */
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
    /* TODO! Use a non-time-symmetric rate matrix for debugging */
    let rate_matrix = rate_matrix_example();
    let distance_threshold = 1e-9 as Float;

    let data_path = if args.len() >= 2 {
        &args[1]
    } else {
        "data/tree_topological.csv"
    };

    let mut record_reader = read_preprocessed_csv(data_path).unwrap();

    let tree;
    let mut distances;
    let sequences_raw;
    (tree, distances, sequences_raw) = deserialize_tree(&mut record_reader).unwrap();

    distances
        .iter_mut()
        .for_each(|d| *d = distance_threshold.max(*d));

    let (num_leaves, seq_length, sequences_2d) =
        try_entry_sequences_from_strings(&sequences_raw).unwrap();

    let num_nodes = tree.len();

    let mut forward_data =
        Vec::<LogTransitionForwardData<{ Entry::DIM }>>::with_capacity(num_nodes);
    /* TODO get rid of Options */
    let mut log_p = Vec::<Option<[Float; Entry::DIM]>>::with_capacity(num_nodes);

    let mut log_likelihood = 0.0 as Float;
    let mut grad_log_prior = [0.0 as Float; Entry::DIM];

    for (column_id, column) in sequences_2d.transpose().column_iter().enumerate() {
        forward_column(
            column.as_view(),
            &tree,
            &distances,
            &mut log_p,
            &mut forward_data,
            rate_matrix.as_view(),
            num_nodes,
            num_leaves,
        );

        let log_p_root = &log_p[num_leaves - 1].unwrap();

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
