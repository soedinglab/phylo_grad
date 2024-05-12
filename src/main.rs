#![allow(dead_code, clippy::needless_range_loop)]
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
use std::fmt::Formatter;

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

struct DisplayArray<'a>(&'a [Float]);
impl<'a> std::fmt::Display for DisplayArray<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        self.0[0].fmt(f)?;
        for value in &self.0[1..] {
            write!(f, ", ")?;
            value.fmt(f)?;
        }
        write!(f, "]")
    }
}

fn rate_matrix_example() -> RateType {
    const DIM: usize = Entry::DIM;
    let rate_matrix_example = -RateType::identity()
        + (1 as Float / (DIM - 1) as Float)
            * (RateType::from_element(1.0 as Float) - RateType::identity());
    rate_matrix_example
}

fn try_residue_sequences_from_strings(
    sequences_raw: &[Option<String>],
) -> Result<na::DMatrix<Residue>, FelsensteinError> {
    let num_leaves = sequences_raw.partition_point(|x| !x.is_none());

    /* TODO remove this line, get the length from the iterator, accept sequences_raw argument as an iterator */
    let seq_expected_length = sequences_raw[0]
        .as_ref()
        .expect("The first node has to be a leaf")
        .len()
        / Entry::CHARS;

    let mut sequences_flat = Vec::<Residue>::with_capacity(num_leaves * seq_expected_length);

    for res in sequences_raw[0..num_leaves]
        .iter()
        .map(|x| match x.as_deref() {
            Some(s) => Ok(s),
            None => Err(FelsensteinError::ORDER),
        })
        .map(|res: Result<&str, FelsensteinError>| {
            process_results(Residue::try_deserialize_string_iter(res?), |it| {
                sequences_flat.extend(it)
            })
        })
    {
        res?
    }
    /* TODO get actual length from iterator, check all lengths are the same */
    let seq_length = sequences_flat.len() / num_leaves;
    let sequences_2d =
        na::DMatrix::from_vec_generic(na::Dyn(seq_length), na::Dyn(num_leaves), sequences_flat)
            .transpose();
    Ok(sequences_2d)
}

fn forward_column<'a>(
    column: impl Iterator<Item = &'a Entry>,
    tree: &[TreeNode],
    //distances: &[Float],
    log_p: &mut Vec<Option<[Float; Entry::DIM]>>,
    forward_data: &ForwardData<{ Entry::DIM }>,
    //rate_matrix: na::SMatrixView<Float, { Entry::DIM }, { Entry::DIM }>,
) {
    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    let num_nodes = tree.len();
    log_p.clear();
    log_p.extend(column.map(|x| Some(Entry::to_log_p(x))));
    let num_leaves = log_p.len();
    log_p.resize(num_nodes, None);

    /* TODO remove copy */
    for i in num_leaves..(num_nodes - 1) {
        let log_p_new = forward_node(i, tree, log_p, forward_data).unwrap();
        log_p[i] = Some(log_p_new);
    }
    let log_p_root = forward_root(num_nodes - 1, tree, log_p, forward_data);
    log_p[num_leaves - 1] = Some(log_p_root);
}

fn test_d_exp() {
    let argument = rate_matrix_example() + RateType::identity();
    let cotangent_vector =
        na::SMatrix::<_, { Entry::DIM }, { Entry::DIM }>::from_element(1.0 as Float);
    let dexp = d_exp_vjp(argument.as_view(), cotangent_vector.as_view());

    println!("d_exp_vjp:");
    let mut tmp_row: na::SVector<Float, { Entry::DIM }>;
    for row in dexp.row_iter() {
        tmp_row = row.transpose().to_owned();
        println!("{:.4}", DisplayArray(tmp_row.as_slice()));
    }

    let tangent_vector =
        na::SMatrix::<_, { Entry::DIM }, { Entry::DIM }>::from_element(1.0 as Float);
    let dexp_jvp = d_exp_jvp(argument.as_view(), tangent_vector.as_view());

    println!("d_exp_jvp:");
    let mut tmp_row: na::SVector<Float, { Entry::DIM }>;
    for row in dexp_jvp.row_iter() {
        tmp_row = row.transpose().to_owned();
        println!("{:.4}", DisplayArray(tmp_row.as_slice()));
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    /* Placeholder values */
    let log_prior = [(Entry::DIM as Float).recip(); Entry::DIM].map(Float::ln);
    /* TODO! Use a non-time-symmetric rate matrix for debugging */
    let rate_matrix = rate_matrix_example();
    let distance_threshold = 1e-9 as Float;
    const COL_LIMIT: ColumnId = 300;

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

    let residue_sequences_2d = try_residue_sequences_from_strings(&sequences_raw).unwrap();

    let (num_leaves, _residue_seq_length) = residue_sequences_2d.shape();
    let num_nodes = tree.len();

    let mut forward_data = ForwardData::<{ Entry::DIM }>::with_capacity(num_nodes);
    /* TODO get rid of Options */
    let mut log_p = Vec::<Option<[Float; Entry::DIM]>>::with_capacity(num_nodes);

    let mut log_likelihood = 0.0 as Float;
    let mut grad_log_prior = [0.0 as Float; Entry::DIM];

    // for (column_id, column) in sequences_2d.column_iter().enumerate() {
    for (column_id, column) in Entry::columns_iter(&residue_sequences_2d).take(COL_LIMIT) {
        /* Right now, this is the same for all columns, but as every column will have its own
        rate matrix, in general we'll have to precompute log_transition for each column */
        forward_data_precompute(&mut forward_data, rate_matrix.as_view(), &distances);

        /* TODO get rid of num_nodes and num_leaves */
        forward_column(
            column.iter(),
            &tree,
            //&distances,
            &mut log_p,
            &forward_data,
            //rate_matrix.as_view(),
        );

        let log_p_root = &log_p[num_leaves - 1].unwrap();

        let mut forward_data_likelihood_lse_arg = [0.0 as Float; Entry::DIM];
        for i in 0..Entry::DIM {
            forward_data_likelihood_lse_arg[i] = log_p_root[i] + log_prior[i];
        }
        let log_likelihood_column = forward_data_likelihood_lse_arg.iter().ln_sum_exp();

        softmax_inplace(&mut forward_data_likelihood_lse_arg);
        let grad_log_prior_column = forward_data_likelihood_lse_arg;

        for i in 0..Entry::DIM {
            grad_log_prior[i] += grad_log_prior_column[i];
        }
        log_likelihood += log_likelihood_column;

        println!(
            "Log likelihood #{:?} = {:.8}",
            column_id, log_likelihood_column
        );
        println!(
            "Gradient of log_prior #{:?} = {:.0}",
            column_id,
            DisplayArray(&grad_log_prior_column)
        );
    }
    println!("Log likelihood = {}", log_likelihood);
    println!(
        "Gradient of log_prior = {:.8}",
        DisplayArray(&grad_log_prior)
    );
}
