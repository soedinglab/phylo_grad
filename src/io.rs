/* https://github.com/rust-lang/rust-by-example/blob/master/src/std_misc/file/read_lines.md */
/* https://docs.rs/csv/latest/csv/cookbook/index.html */
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::data_types::*;
use crate::tree::TreeNode;
use ::itertools::{multiunzip, process_results};
use serde::Deserialize;

/* TODO:
1. Collect residues into (Vec<Entry>, row_length) instead of Vec<Vec<Entry>>.
2. Collect residues into ndarray::Array2 or similar. */

/* TODO #[serde(flatten)] */
#[derive(Debug, Deserialize)]
struct InputRecord {
    parent: usize,
    left: Option<usize>,
    right: Option<usize>,
    distance: Option<Float>,
    sequence: Option<String>,
}
/* TODO check is_ascii() */
type InputTuple = (TreeNode, Float, Option<String>);

impl From<InputRecord> for InputTuple {
    fn from(input: InputRecord) -> Self {
        let distance: Float = match input.distance {
            Some(d) => d,
            None => -1.0,
        };

        (
            TreeNode {
                parent: input.parent,
                left: input.left,
                right: input.right,
            },
            distance,
            input.sequence,
        )
    }
}

pub fn read_preprocessed_csv<P>(filename: P) -> Result<csv::Reader<BufReader<File>>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let bufreader = BufReader::new(file);
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bufreader);
    Ok(rdr)
}

pub fn deserialize_tree(
    reader: &mut csv::Reader<BufReader<File>>,
) -> Result<(Vec<TreeNode>, Vec<Float>, Vec<Option<String>>), Box<dyn Error>> {
    let (tree, distances, sequences) =
        process_results(reader.deserialize::<InputRecord>(), |record| {
            multiunzip(record.map(InputTuple::from))
        })?;
    Ok((tree, distances, sequences))
}
