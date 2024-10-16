import csv

import Bio.AlignIO as AlignIO
import Bio.Phylo as Phylo
from Bio.Align import MultipleSeqAlignment
import torch

amino_mapping = {'A' : 0, 'R' : 1, 'N' : 2, 'D' : 3, 'C' : 4, 'Q' : 5, 'E' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'L' : 10, 'K' : 11, 'M' : 12, 'F' : 13, 'P' : 14, 'S' : 15, 'T' : 16, 'W' : 17, 'Y' : 18, 'V' : 19, '-' : 20} 
# Hard mapping from each possible character to a number between 0 and 4
nuc_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, '.': 4, 'R': 2,
               'Y': 3, 'S': 2, 'W': 3, 'K': 3, 'M': 0, 'B': 3, 'D': 3, 'H': 3, 'V': 0, 'N': 4}

def read_tree_file_seq(tree: str):
    with open(tree, 'r') as f:
        L = int(f.readline())
        reader = csv.reader(f)
        return [(int(parent), None if dist == '' else float(dist), None if seq == '' else seq) for [parent, dist, seq] in reader], L

def seq_to_onehot(seq: str | None) -> torch.Tensor | None:
    if seq is None:
        return None
    onehot = torch.full((len(seq), 5), -1000)
    for i, c in enumerate(seq):
        onehot[i, nuc_mapping[c]] = 0
    return onehot
        
def read_tree_file(tree: str):
    tree, L = read_tree_file_seq(tree)
    tree = [(p, t, seq_to_onehot(seq)) for p, t, seq in tree]
    return tree