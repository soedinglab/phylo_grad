/* https://github.com/rust-lang/rust-by-example/blob/master/src/std_misc/file/read_lines.md */
/* https://docs.rs/csv/latest/csv/cookbook/index.html */
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::data_types::*;
use serde::Deserialize;

/* TODO f64 or Float? */
#[derive(Debug, Deserialize)]
pub struct RawInputRecord {
    #[serde(rename = "0")]
    pub parent: i64,
    #[serde(rename = "1")]
    pub distance: Option<Float>,
    #[serde(rename = "2")]
    pub sequence: Option<String>,
}

pub fn read_raw_csv<P>(
    filename: P,
    skip_lines: usize,
) -> Result<csv::Reader<BufReader<File>>, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let mut bufreader = BufReader::new(file);

    /* TODO how to avoid saving useless lines?
    1. bufreader.lines().nth() */
    for _ in 0..skip_lines {
        let mut _line = String::new();
        bufreader.read_line(&mut _line)?;
    }

    let rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bufreader);
    Ok(rdr)
}

pub fn deserialize_raw_tree<R>(
    reader: &mut csv::Reader<R>,
) -> Result<Vec<RawInputRecord>, csv::Error>
where
    R: std::io::Read,
{
    reader
        .deserialize::<RawInputRecord>()
        .collect::<Result<Vec<RawInputRecord>, csv::Error>>()
}
