"""
    This module contains functions to read a newick file and put it in the required format for phylo_grad
"""

import Bio.AlignIO as AlignIO
import Bio.Phylo as Phylo
from Bio.Align import MultipleSeqAlignment
import torch

amino_mapping = {'A' : 0, 'R' : 1, 'N' : 2, 'D' : 3, 'C' : 4, 'Q' : 5, 'E' : 6, 'G' : 7, 'H' : 8, 'I' : 9, 'L' : 10, 'K' : 11, 'M' : 12, 'F' : 13, 'P' : 14, 'S' : 15, 'T' : 16, 'W' : 17, 'Y' : 18, 'V' : 19, '-' : 20, '.' : 20} 

def read_newick(newick_file: str):
    """
        Reads a newick file and returns a list of tuples with the parent id and branch length of each node.
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

    parent_list = [(node.parent_id, node.branch_length)
                   for node in nodes]

    return parent_list, num_leaf

def amino_to_embedding(seq: str | None) -> torch.Tensor | None:
    if seq is None:
        return None
    onehot = torch.full((len(seq), 20), -float('inf'), dtype=torch.float64)
    for i, c in enumerate(seq):
        amino = amino_mapping[c]
        if amino < 20:
            onehot[i,amino] = 0
        else:
            onehot[i] = torch.zeros(20, dtype=torch.float64)
            
    return onehot
    
def read_newick_fasta(newick_file: str, fasta_file: str):
    alignment = AlignIO.read(fasta_file, "fasta")
    assert isinstance(alignment, MultipleSeqAlignment)

    tree = Phylo.read(newick_file, "newick")
    assert isinstance(tree, Phylo.BaseTree.Tree)

    seq_dict = {seq.id: seq.seq for seq in alignment}

    for node in tree.get_terminals():
        node.sequence = seq_dict[node.name]
    for node in tree.get_nonterminals():
        node.sequence = None

    nodes = tree.get_terminals() + tree.get_nonterminals()

    node_ids = {node: id for id, node in enumerate(nodes)}
    node_ids[None] = -1

    def traverse(node, parent=None):
        node.parent_id = node_ids[parent]
        for child in node.clades:
            traverse(child, node)

    traverse(tree.root)

    parent_list = [(node.parent_id, node.branch_length, node.sequence)
                   for node in nodes]

    return parent_list, alignment.get_alignment_length()