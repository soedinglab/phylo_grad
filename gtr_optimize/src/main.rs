use std::collections::HashMap;

use seq_io::fasta::Record;

/// Simple command line interface for gtr_optimize library
/// # Arguments
/// * `args[1]` - Newick file path
/// * `args[2]` - Fasta file path
/// * `args[3]` - "global" or "local" to select optimization mode
pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let newick = std::fs::read_to_string(&args[1]).expect("Could not read newick file");
    let mut alignment = seq_io::fasta::Reader::new(std::fs::File::open(&args[2]).expect("Could not open fasta file"));
    let mut sequences = HashMap::new();
    while let Some(result) = alignment.next() {
        let record = result.expect("Error reading fasta record");
        let seq: Vec<u8> = record.seq().iter().copied().collect();
        sequences.insert(record.id().unwrap().to_string(), seq);
    }

    let ll = if args[3] == "global" {
        gtr_optimize::optimize_gtr_global(&newick, &sequences)
    } else if args[3] == "local" {
        gtr_optimize::optimize_gtr_local(&newick, &sequences)
    } else {
        panic!("Third argument must be 'global' or 'local'");
    };

    println!("BEST SCORE FOUND : {}", ll);
}