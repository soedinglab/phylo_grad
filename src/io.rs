/* https://github.com/rust-lang/rust-by-example/blob/master/src/std_misc/file/read_lines.md */
/* https://docs.rs/csv/latest/csv/cookbook/index.html */
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::data_types::*;
use crate::tree::TreeNode;
use serde::Deserialize;
use std::convert::TryFrom;

/* TODO:
1. Collect residues into (Vec<Entry>, row_length) instead of Vec<Vec<Entry>>.
2. Collect residues into ndarray::Array2 or similar. */

/* TODO #[serde(flatten)] */
#[derive(Debug, Deserialize)]
pub struct PreprocessedRecord {
    parent: usize,
    left: Option<usize>,
    right: Option<usize>,
    distance: Option<Float>,
    sequence: Option<String>,
}

pub type InputTuple = (TreeNode, Vec<ResidueExtended>);

impl From<PreprocessedRecord> for InputTuple {
    fn from(input: PreprocessedRecord) -> Self {
        let enum_sequence = match input.sequence {
            Some(str) => str
                .chars()
                .map(ResidueExtended::from)
                .collect::<Vec<ResidueExtended>>(),
            None => Vec::<ResidueExtended>::new(),
        };

        let distance: Float = match input.distance {
            Some(d) => d,
            None => -1.0,
        };

        (
            TreeNode {
                parent: input.parent,
                left: input.left,
                right: input.right,
                distance: distance,
            },
            enum_sequence,
        )
    }
}

pub fn read_preprocessed_csv<P>(filename: P) -> Result<csv::Reader<BufReader<File>>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let mut bufreader = BufReader::new(file);
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bufreader);
    Ok(rdr)
}

pub fn deserialize_tree(
    reader: &mut csv::Reader<BufReader<File>>,
) -> (Vec<TreeNode>, Vec<Vec<ResidueExtended>>) {
    let (tree, sequences): (Vec<TreeNode>, Vec<Vec<ResidueExtended>>) = reader
        .deserialize::<PreprocessedRecord>()
        .map(|x| x.unwrap())
        .map(InputTuple::from)
        .unzip();
    (tree, sequences)
}
