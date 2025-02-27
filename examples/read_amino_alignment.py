import phylo_grad
import numpy as np
from Bio import AlignIO

amino_mapping = {'A' : 0, 'R' : 1, 'N' : 2, 'D' : 3, 'C' : 4, 'Q' : 5, 'E' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'L' : 10, 'K' : 11, 'M' : 12, 'F' : 13, 'P' : 14, 'S' : 15, 'T' : 16, 'W' : 17, 'Y' : 18, 'V' : 19, '-' : 20}

def str2logp(seq : str) -> np.ndarray:
    num = np.array([amino_mapping[aa] for aa in seq])
    probs = np.eye(20)
    # '-' has all probabilities set to 1 to ignore it in the likelihood
    probs = np.vstack((probs, np.ones(20)))
    with np.errstate(divide='ignore'):
        return np.log(probs[num])

def fasta2dict(fasta_file : str) -> dict:
    alignment = AlignIO.read(fasta_file, "fasta")
    return {seq.id: str2logp(seq.seq) for seq in alignment}

tree = phylo_grad.FelsensteinTree.from_newick('tree.newick', fasta2dict('alignment.fasta'))
