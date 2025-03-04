import io

from . import _phylo_grad
import numpy as np
from Bio import Phylo
from Bio import AlignIO

class FelsensteinTree:
    def __init__(self, tree, leaf_log_p, distance_threshold=1e-4):
        assert isinstance(tree, np.ndarray)
        assert isinstance(leaf_log_p, np.ndarray)
        assert isinstance(distance_threshold, float)
        
        assert len(leaf_log_p.shape) == 3, "leaf_log_p must have shape (L, N, DIM)"
        
        dim = leaf_log_p.shape[2]
        if leaf_log_p.dtype == np.float32:
            dtype = 'f32'
        elif leaf_log_p.dtype == np.float64:
            dtype = 'f64'
        else:
            raise ValueError('leaf_log_p must be either np.float32 or np.float64')

        assert tree.dtype == leaf_log_p.dtype, "tree and leaf_log_p must have the same dtype"
        
        try:
            tree_class = getattr(_phylo_grad, f'Backend_{dtype}_{dim}')
        except AttributeError:
            raise ValueError(f'Unsupported dim {dim}, see Readme.md for more information')
        
        self.dtype = leaf_log_p.dtype
        self.tree = tree_class(tree, leaf_log_p, distance_threshold)
        self.L = leaf_log_p.shape[0]
        
    @classmethod
    def from_newick(cls, newick : str | io.TextIOBase, leaf_log_p_dict : dict, dtype: np.floating = np.float64, distance_threshold : float = 1e-4):
        """
            newick: File Handle or File name containing the newick tree
            leaf_log_p_dict: A dictionary mapping leaf names to log probabilities, {seq_id : np array with dimensions [seq_length, 20]}
            dtype: The desired dtype, either np.float32 or np.float64, leaf_log_p will be converted to this dtype
            distance_threshold: Every edge of the tree will set to be at least that long because very small edge lengts can lead to numerical unstabilities.
        """
        tree = Phylo.read(newick, "newick")
        
        nodes = tree.get_terminals() + tree.get_nonterminals()

        node_ids = {node: id for id, node in enumerate(nodes)}
        node_ids[None] = -1

        def traverse(node, parent=None):
            node.parent_id = node_ids[parent]
            for child in node.clades:
                traverse(child, node)

        traverse(tree.root)

        parent_list = []
        leaf_log_p = []
        
        for node in nodes:
            parent_list.append((node.parent_id, node.branch_length))
            if node.is_terminal():
                leaf_log_p.append(leaf_log_p_dict[node.name])
        
        leaf_log_p_array = np.array(leaf_log_p, dtype=dtype).transpose(1, 0, 2)
        
        return cls(np.array(parent_list, dtype=dtype), leaf_log_p_array, distance_threshold)
        
        
    def calculate_gradients(self, S, sqrt_pi):
        assert isinstance(S, np.ndarray), "S must be a numpy array"
        assert isinstance(sqrt_pi, np.ndarray), "sqrt_pi must be a numpy array"
        assert S.dtype == self.dtype, "S must have the same dtype as the tree"
        assert sqrt_pi.dtype == self.dtype, "sqrt_pi must have the same dtype as the tree"
        return self.tree.calculate_gradients(S, sqrt_pi)

    def get_sequence_length(self) -> int:
        return self.L
