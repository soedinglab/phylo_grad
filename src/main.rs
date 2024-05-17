#![allow(clippy::needless_range_loop)]
extern crate nalgebra as na;
/* TODO pyo3 */

use itertools::{process_results, Itertools};
use logsumexp::LogSumExp;
use rayon::prelude::*;

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
    Residue: EntryTrait,
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

fn forward_column<const DIM: usize, Entry>(
    column: impl Iterator<Item = Entry>,
    tree: &[TreeNode],
    log_p: &mut Vec<Option<[Float; DIM]>>,
    forward_data: &ForwardData<DIM>,
) where
    Entry: EntryTrait<LogPType = [Float; DIM]>,
{
    /* Compared to collect(), this reduces the # of allocation calls
    but increases peak memory usage; investigate */
    let num_nodes = tree.len();
    log_p.clear();
    log_p.extend(column.map(|x| Some(Entry::to_log_p(&x))));
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

    let (num_leaves, residue_seq_length) = residue_sequences_2d.shape();
    let num_nodes = tree.len();

    let index_pairs: Vec<(_, _)> = (0..residue_seq_length)
        .tuple_combinations::<(_, _)>()
        .take(COL_LIMIT)
        .collect();

    let log_likelihood_total: Vec<Float>;
    let grad_rate_total: Vec<na::SMatrix<Float, { Entry::DIM }, { Entry::DIM }>>;
    let grad_log_prior_total: Vec<[Float; Entry::DIM]>;

    (
        log_likelihood_total,
        (grad_rate_total, grad_log_prior_total),
    ) = index_pairs
        .par_iter()
        .map(|column_id| {
            /* State storage */
            let mut forward_data = ForwardData::<{ Entry::DIM }>::with_capacity(num_nodes);
            /* TODO get rid of Options */
            let mut log_p = Vec::<Option<[Float; Entry::DIM]>>::with_capacity(num_nodes);

            let mut backward_data =
                Vec::<BackwardData<{ Entry::DIM }>>::with_capacity(num_nodes - num_leaves);

            /* State storage end */

            let (left_id, right_id) = *column_id;
            let left_half = residue_sequences_2d.column(left_id);
            let right_half = residue_sequences_2d.column(right_id);
            let column_iter = std::iter::zip(left_half.iter(), right_half.iter()).map(
                |(left_residue, right_residue)| {
                    ResiduePair::<Residue>(*left_residue, *right_residue)
                },
            );

            /* Right now, this is the same for all columns, but as every column will have its own
            rate matrix, in general we'll have to precompute log_transition for each column */
            forward_data_precompute(&mut forward_data, rate_matrix.as_view(), &distances);

            forward_column(column_iter, &tree, &mut log_p, &forward_data);

            let log_p_root = &log_p[num_leaves - 1].unwrap();

            let mut forward_data_likelihood_lse_arg = [0.0 as Float; Entry::DIM];
            for i in 0..Entry::DIM {
                forward_data_likelihood_lse_arg[i] = log_p_root[i] + log_p_prior[i];
            }
            let log_likelihood_column = forward_data_likelihood_lse_arg.iter().ln_sum_exp();

            softmax_inplace(&mut forward_data_likelihood_lse_arg);
            let grad_log_prior_column = forward_data_likelihood_lse_arg;
            let grad_log_p_root = forward_data_likelihood_lse_arg;

            /* Notice that child_input values are always added, so the log_p input for children is always the same.
            We will therefore store their common grad_log_p in the parent node's BackwardData. */
            /* TODO: it is possible to free grad_log_p's for the previous tree level. */
            let mut grad_rate_column =
                na::SMatrix::<Float, { Entry::DIM }, { Entry::DIM }>::from_element(0.0 as Float);
            /* root.backward */
            backward_data.push(BackwardData {
                grad_log_p: grad_log_p_root,
            });
            /* node.backward for non-terminal nodes */
            for id in (num_leaves..num_nodes - 1).rev() {
                let parent_id = &tree[id].parent;
                let parent_backward_id = num_nodes - parent_id - 1;
                let grad_log_p_input = backward_data[parent_backward_id].grad_log_p;
                let log_p_input = &log_p[id].unwrap();
                let distance_current = distances[id];
                let fwd_data_current = &forward_data.log_transition[id];
                let (grad_rate, grad_log_p) = d_child_input_vjp(
                    grad_log_p_input,
                    distance_current,
                    log_p_input,
                    fwd_data_current,
                    true,
                );
                grad_rate_column += grad_rate;
                backward_data.push(BackwardData {
                    grad_log_p: grad_log_p.unwrap(),
                });
            }
            /* For leaves, we only compute grad_rate */
            for id in (0..num_leaves).rev() {
                let parent_id = &tree[id].parent;
                let parent_backward_id = num_nodes - parent_id - 1;
                let grad_log_p_input = backward_data[parent_backward_id].grad_log_p;
                let log_p_input = &log_p[id].unwrap();
                let distance_current = distances[id];
                let fwd_data_current = &forward_data.log_transition[id];
                let (grad_rate, _) = d_child_input_vjp(
                    grad_log_p_input,
                    distance_current,
                    log_p_input,
                    fwd_data_current,
                    false,
                );
                grad_rate_column += grad_rate;
            }

            backward_data.clear();

            grad_rate_column.transpose_mut();

            (
                log_likelihood_column,
                (grad_rate_column, grad_log_prior_column),
            )
        })
        .unzip();
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
