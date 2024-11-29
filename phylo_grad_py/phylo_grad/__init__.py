from . import _phylo_grad
import numpy as np
from Bio import Phylo
from Bio import AlignIO

class FelsensteinTree:
    def __init__(self, tree, leaf_log_p, distance_threshold):
        assert(isinstance(tree, np.ndarray))
        assert(isinstance(leaf_log_p, np.ndarray))
        assert(isinstance(distance_threshold, float))
        
        assert(len(leaf_log_p.shape) == 3)
        
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
            raise ValueError(f'Unsupported dim {dim}')
        
        self.tree = tree_class(tree, leaf_log_p, distance_threshold)
        
    @classmethod
    def from_newick(cls, newick, leaf_log_p_dict, distance_threshold, dtype):
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
        
        leaf_log_p_array = np.array(leaf_log_p, dtype=dtype)
        
        return cls(np.array(parent_list, dtype=dtype), leaf_log_p_array, distance_threshold)
        
        
    def calculate_gradients(self, S, sqrt_pi):
        return self.tree.calculate_gradients(S, sqrt_pi)