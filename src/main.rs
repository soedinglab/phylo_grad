#![allow(clippy::needless_range_loop)]
extern crate nalgebra as na;
/* TODO pyo3 */

use itertools::Itertools;

use std::fmt::Formatter;

mod backward;
mod data_types;
mod forward;
mod io;
mod preprocessing;
mod train;
mod tree;

use crate::data_types::*;
use crate::io::*;
use crate::preprocessing::*;
use crate::train::*;

struct DisplayArray<'a, T>(&'a [T])
where
    T: std::fmt::Display;
impl<'a, T> std::fmt::Display for DisplayArray<'a, T>
where
    T: std::fmt::Display,
{
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

fn rate_matrix_example<const DIM: usize>() -> na::SMatrix<Float, DIM, DIM> {
    let rate_matrix_example = -na::SMatrix::<Float, DIM, DIM>::identity()
        + (1 as Float / (DIM - 1) as Float)
            * (na::SMatrix::<Float, DIM, DIM>::from_element(1.0 as Float)
                - na::SMatrix::<Float, DIM, DIM>::identity());
    rate_matrix_example
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    /* Placeholder values */
    let log_p_prior = [(Entry::DIM as Float).recip(); Entry::DIM].map(Float::ln);
    /* TODO! Use a non-time-symmetric rate matrix for debugging */
    let rate_matrix = rate_matrix_example::<{ Entry::DIM }>();
    let distance_threshold = 1e-4 as Float;
    const COL_LIMIT: usize = 3;

    let raw_data_path = if args.len() >= 2 {
        &args[1]
    } else {
        "data/tree.txt"
    };

    let (tree, distances, residue_sequences_2d) = {
        let mut record_reader = read_raw_csv(raw_data_path, 1).unwrap();
        let raw_tree = deserialize_raw_tree(&mut record_reader).unwrap();

        let tree;
        let mut distances;
        let sequences_raw;
        (tree, distances, sequences_raw) = preprocess_weak(&raw_tree).unwrap();

        distances
            .iter_mut()
            .for_each(|d| *d = distance_threshold.max(*d));

        let residue_sequences_2d = try_residue_sequences_from_strings(&sequences_raw).unwrap();

        (tree, distances, residue_sequences_2d)
    };

    let (_num_leaves, residue_seq_length) = residue_sequences_2d.shape();

    let index_pairs: Vec<(_, _)> = (0..residue_seq_length)
        .tuple_combinations::<(_, _)>()
        .take(COL_LIMIT)
        .collect();

    let n_columns = index_pairs.len();

    let rate_matrices: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>> =
        std::iter::repeat(rate_matrix).take(n_columns).collect();
    let log_p_priors: Vec<[Float; Entry::DIM]> =
        std::iter::repeat(log_p_prior).take(n_columns).collect();

    let log_likelihood_total: Vec<Float>;
    let grad_rate_total: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>>;
    let grad_log_prior_total: Vec<[Float; Entry::DIM]>;

    (log_likelihood_total, grad_rate_total, grad_log_prior_total) = train_parallel(
        &index_pairs,
        residue_sequences_2d.as_view(),
        /* TODO as_deref() */
        &rate_matrices,
        &log_p_priors,
        &tree,
        &distances,
    );

    for (column_id, log_likelihood_column) in
        std::iter::zip(index_pairs.iter(), log_likelihood_total.iter())
    {
        println!(
            "Log likelihood #{:?} = {:.8}",
            column_id, log_likelihood_column
        );
    }
    for (column_id, grad_log_prior_column) in
        std::iter::zip(index_pairs.iter(), grad_log_prior_total.iter())
    {
        println!(
            "Gradient of log_prior #{:?} = {:.10}",
            column_id,
            DisplayArray(grad_log_prior_column)
        );
    }
    for (column_id, grad_rate_column) in std::iter::zip(index_pairs.iter(), grad_rate_total.iter())
    {
        println!("Gradient of rate #{:?}:", column_id);
        for row in grad_rate_column.row_iter() {
            let tmp_row = row.transpose().to_owned();
            println!("{:12.8}", DisplayArray(tmp_row.as_slice()));
        }
    }
}
