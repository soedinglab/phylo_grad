use std::collections::HashMap;

use seq_io::fasta::Record;

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

    let _optimized_newick = phylo_grad::tree_opt::optimize_tree(&newick, &sequences);
}