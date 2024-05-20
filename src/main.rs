#![allow(clippy::needless_range_loop)]
extern crate nalgebra as na;
/* TODO pyo3 */

use itertools::{process_results, Itertools};

use std::fmt::Formatter;

mod backward;
mod data_types;
mod forward;
mod io;
mod train;
mod tree;

use crate::data_types::*;
use crate::io::*;
use crate::train::*;

impl FelsensteinError {
    pub const ORDER: Self = Self::DeserializationError("The tree is not topoligically ordered");
}

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

fn try_residue_sequences_from_strings<Residue>(
    sequences_raw: &[Option<String>],
) -> Result<na::DMatrix<Residue>, FelsensteinError>
where
    Residue: ResidueTrait,
{
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
    let sequences_2d = na::DMatrix::from_vec_storage(na::VecStorage::new(
        na::Dyn(seq_length),
        na::Dyn(num_leaves),
        sequences_flat,
    ))
    .transpose();
    Ok(sequences_2d)
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    /* Placeholder values */
    let log_p_prior = [(Entry::DIM as Float).recip(); Entry::DIM].map(Float::ln);
    /* TODO! Use a non-time-symmetric rate matrix for debugging */
    let rate_matrix = rate_matrix_example::<{ Entry::DIM }>();
    let distance_threshold = 1e-4 as Float;
    const COL_LIMIT: ColumnId = 1_000_000;

    let data_path = if args.len() >= 2 {
        &args[1]
    } else {
        "data/tree_topological.csv"
    };
    let (tree, distances, residue_sequences_2d) = {
        let mut record_reader = read_preprocessed_csv(data_path).unwrap();

        let tree;
        let mut distances;
        let sequences_raw;
        (tree, distances, sequences_raw) = deserialize_tree(&mut record_reader).unwrap();

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

    let log_likelihood_total: Vec<Float>;
    let grad_rate_total: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>>;
    let grad_log_prior_total: Vec<[Float; Entry::DIM]>;

    (log_likelihood_total, grad_rate_total, grad_log_prior_total) = train_parallel(
        &index_pairs,
        residue_sequences_2d.as_view(),
        rate_matrix.as_view(),
        &log_p_prior,
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
            "Gradient of log_prior #{:?} = {:.8}",
            column_id,
            DisplayArray(grad_log_prior_column)
        );
    }
    for (column_id, grad_rate_column) in std::iter::zip(index_pairs.iter(), grad_rate_total.iter())
    {
        println!("Gradient of rate #{:?}:", column_id);
        let mut tmp_row: na::SVector<Float, { Entry::DIM }>;
        for row in grad_rate_column.row_iter() {
            tmp_row = row.transpose().to_owned();
            println!("{:12.8}", DisplayArray(tmp_row.as_slice()));
        }
    }
}
