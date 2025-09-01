"""
    This module contains functions to read a newick file and put it in the required format for phylo_grad
"""

import Bio.AlignIO as AlignIO
import Bio.Phylo as Phylo
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

def read_newick(newick_file: str) -> dict:
    """
        Reads a newick file and returns a parent_list, branch_lengths and number of leaf nodes.
    """
    
    tree = Phylo.read(newick_file, "newick")
    assert isinstance(tree, Phylo.BaseTree.Tree)
    
    num_leaf = len(tree.get_terminals())

    nodes = tree.get_terminals() + tree.get_nonterminals()

    node_ids = {node: id for id, node in enumerate(nodes)}
    node_ids[None] = -1

    def traverse(node, parent=None):
        node.parent_id = node_ids[parent]
        for child in node.clades:
            traverse(child, node)

    traverse(tree.root)

    parent_list = [node.parent_id for node in nodes]
    branch_lengths = [node.branch_length for node in nodes]

    return {
        'parent_list': parent_list,
        'branch_lengths': branch_lengths,
        'num_leaf': num_leaf
    }