/* https://github.com/rust-lang/rust-by-example/blob/master/src/std_misc/file/read_lines.md */
/* https://docs.rs/csv/latest/csv/cookbook/index.html */
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/* For fixing the tree */
use crate::data_types::*;
use crate::tree_iterators_rs::prelude::*;
use serde::Deserialize;
use std::convert::TryFrom;

#[derive(Debug, Deserialize)]
pub struct RecordTuple(Option<i32>, Option<Float>, Option<String>);

#[derive(Debug)]
pub struct InputTreeNode {
    parent: Option<usize>,
    distance: Option<Float>,
    left: Option<usize>,
    right: Option<usize>,
    sequence: Option<Vec<ResidueExtended>>,
}

impl From<RecordTuple> for InputTreeNode {
    fn from(value: RecordTuple) -> Self {
        let sequence = match value.2 {
            Some(str) => Some(
                str.chars()
                    .map(ResidueExtended::from)
                    .collect::<Vec<ResidueExtended>>(),
            ),
            None => None,
        };

        let parent = match value.0 {
            Some(integer) => {
                if integer >= 0 {
                    Some(integer as usize)
                } else {
                    None
                }
            }
            None => None,
        };

        Self {
            parent: parent,
            distance: value.1,
            left: None,
            right: None,
            sequence: sequence,
        }
    }
}

pub fn binary_tree_from_file<P>(
    filename: P,
) -> Result<BinaryTreeNode<InputTreeNode>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let root = BinaryTreeNode {
        value: InputTreeNode {
            parent: None,
            distance: None,
            left: None,
            right: None,
            sequence: None,
        },
        left: None,
        right: None,
    };

    let mut record_reader = read_csv(filename).unwrap();
    let mut input = record_reader
        .deserialize::<RecordTuple>()
        .map(|x| match x {
            Ok(rt) => Ok(InputTreeNode::from(rt)),
            Err(E) => Err(E),
        })
        .collect::<Result<Vec<InputTreeNode>, _>>();
    for itn in input? {
        let parent = itn.parent;
    }
    Ok(root)
}

/* Goal:
- Read the first line
- Parse the remaining lines as csv
- Create a TreeFixed instance, a distance array and a 2d sequence array of enums
- - Fill the child pointers
- - - Handle the root node having three children
*/

/* pub fn transform_to_postorder<P>(in_path: P, out_path: P) -> Result<(), Box<dyn Error>>
where
    P: AsRef<Path>,
{

}
*/

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
