"""
    This module contains functions to read a newick file and put it in the required format for phylo_grad
"""

import Bio.AlignIO as AlignIO
from Bio.Align import MultipleSeqAlignment
import numpy as np

amino_mapping = {'A' : 0, 'R' : 1, 'N' : 2, 'D' : 3, 'C' : 4, 'Q' : 5, 'E' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'L' : 10, 'K' : 11, 'M' : 12, 'F' : 13, 'P' : 14, 'S' : 15, 'T' : 16, 'W' : 17, 'Y' : 18, 'V' : 19, '-' : 20, '.' : 20} 

def amino_to_embedding(seq: str) -> np.ndarray:
    onehot = np.full((len(seq), 20), -float('inf'), dtype=np.float64)
    for i, c in enumerate(seq):
        amino = amino_mapping[c]
        if amino < 20:
            onehot[i,amino] = 0
        else:
            onehot[i] = np.zeros(20, dtype=np.float64)

    return onehot
    
def read_fasta(fasta_file: str) -> dict:
    alignment = AlignIO.read(fasta_file, "fasta")
    assert isinstance(alignment, MultipleSeqAlignment)

    seq_dict = {seq.id: amino_to_embedding(seq.seq) for seq in alignment}

    return seq_dict