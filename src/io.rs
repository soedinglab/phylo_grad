/* https://github.com/rust-lang/rust-by-example/blob/master/src/std_misc/file/read_lines.md */
/* https://docs.rs/csv/latest/csv/cookbook/index.html */
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/* For fixing the tree */
use crate::data_types::*;
use crate::tree::TreeNode;
use serde::Deserialize;
use std::convert::TryFrom;

/* TODO #[serde(flatten)] */

#[derive(Debug, Deserialize)]
pub struct PreprocessedRecord {
    parent: usize,
    left: Option<usize>,
    right: Option<usize>,
    distance: Option<Float>,
    sequence: Option<String>,
}

pub type InputTuple = (TreeNode, Option<Vec<ResidueExtended>>);

impl From<PreprocessedRecord> for InputTuple {
    fn from(input: PreprocessedRecord) -> Self {
        let enum_sequence = match input.sequence {
            Some(str) => Some(
                str.chars()
                    .map(ResidueExtended::from)
                    .collect::<Vec<ResidueExtended>>(),
            ),
            None => None,
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

/* TODO remove */
#[derive(Debug, Deserialize)]
pub struct RecordTuple(Option<i32>, Option<Float>, Option<String>);

pub fn read_csv<P>(filename: P) -> Result<csv::Reader<BufReader<File>>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let mut bufreader = BufReader::new(file);

    let mut s = String::new();
    let mut sequence_length: Option<usize> = None;
    bufreader.read_line(&mut s)?;
    if let Ok(num) = s.trim().parse::<usize>() {
        sequence_length = Some(num);
    }
    //println!("{sequence_length}");

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bufreader);
    Ok(rdr)
}
