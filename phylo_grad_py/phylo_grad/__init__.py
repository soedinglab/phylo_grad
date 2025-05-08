import io

from . import _phylo_grad
import numpy as np
from Bio import Phylo

class FelsensteinTree:
    """A class to represent a phylogenetic tree, it contains the tree topology and edge lengths as well as the log probabilities of the leaf nodes"""
    def __init__(self, tree, leaf_log_p, distance_threshold=1e-4, gpu = False):
        """Typically you should prefer to use the from_newick method to create a FelsensteinTree object.
        For more information see the Rust documentation of phylo_grad
        """
        assert isinstance(tree, np.ndarray)
        assert isinstance(leaf_log_p, np.ndarray)
        assert isinstance(distance_threshold, float)
        
        assert len(leaf_log_p.shape) == 3, "leaf_log_p must have shape [L, N, DIM]"
        
        dim = leaf_log_p.shape[2]
        if leaf_log_p.dtype == np.float32:
            dtype = 'f32'
        elif leaf_log_p.dtype == np.float64:
            dtype = 'f64'
        else:
            raise ValueError('leaf_log_p must be either np.float32 or np.float64')

        assert tree.dtype == leaf_log_p.dtype, "tree and leaf_log_p must have the same dtype"
        
        self.dtype = leaf_log_p.dtype
        self.L = leaf_log_p.shape[0]
        self.dim = dim
        
        if gpu:
            import phylo_grad_gpu
            self.tree = phylo_grad_gpu.FelsensteinTree(tree, leaf_log_p, distance_threshold)
        else:
            try:
                tree_class = getattr(_phylo_grad, f'Backend_{dtype}_{dim}')
            except AttributeError:
                raise ValueError(f'Unsupported dim {dim}, see Readme.md for more information')
            
            self.tree = tree_class(tree, leaf_log_p, distance_threshold)
        
    @classmethod
    def from_newick(cls, newick : str | io.TextIOBase, leaf_log_p_dict : dict, dtype: np.floating = np.float64, distance_threshold : float = 1e-4, gpu = False) -> 'FelsensteinTree':
        """
            Preferred method to create a FelsensteinTree object.
            
            Each of the leaf nodes in the tree needs an array of shape [L, DIM] where L is the sequence length and DIM is number of states (20 for amino acids, 4 for nucleotides, other custom states).
            The entries represent the log probabilities of the states at that position.
            For example with the 4 states A=0, C=1, G=2, T=3, the log probabilities array of the sequence "ACGT" would be:
            [[0.0, -inf, -inf, -inf], # A
            [-inf, 0.0, -inf, -inf], # C
            [-inf, -inf, 0.0, -inf], # G
            [-inf, -inf, -inf, 0.0]] # T
            A gap can be represented by [0.0,0.0,0.0,0.0] meaning all possible states have a probability of 1.0 resulting in ignoring the gap in the likelihood calculation.
            
            :param str newick: File handle or File name containing a tree in newick format
            :param dict leaf_log_p_dict: A dictionary mapping leaf names to log probabilities, {seq_id : log_p_array}
            :param dtype: The desired dtype, either np.float32 or np.float64, arrays in leaf_log_p will be converted to this dtype
            :param float distance_threshold: Every edge of the tree will set to be at least that long because very small edge lengths can lead to numerical unstabilities.
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
        
        return cls(np.array(parent_list, dtype=dtype), leaf_log_p_array, distance_threshold, gpu)
        
        
    def calculate_gradients(self, S : np.ndarray, sqrt_pi : np.ndarray) -> dict:
        """Calculates the gradients of the log likelihood with respect to the given substitution model.
        
        Details of the parametrization of the substitution model can be found in the PhyloGrad paper.
        
        In case one of the symmetric matrices are not diagonalizable, the gradients will be 0.0 and the log likelihood will be -inf.
        This only happens when the matrix has very big or very small eigenvalues (absolute value bigger than 1e5).

        :param S: The symmetric S matrix of the substitution model, shape [L, DIM, DIM]
        :param sqrt_pi: The square root of the stationary distribution, shape [L, DIM]
        
        :return: dict with the keys 'grad_s', 'grad_sqrt_pi' and 'log_likelihood'
                 'grad_s' has the same shape as S, 'grad_sqrt_pi' has the same shape as sqrt_pi and are the gradients
                 'log_likelihood' has shape [L] and is the log likelihood of each column given the substitution model
                 
        """
        assert isinstance(S, np.ndarray), "S must be a numpy array"
        assert S.shape == (self.L, self.dim, self.dim) or S.shape == (1, self.dim, self.dim), "S must have shape [L, DIM, DIM] or [1, DIM, DIM]"
        assert isinstance(sqrt_pi, np.ndarray), "sqrt_pi must be a numpy array"
        assert sqrt_pi.shape == (self.L, self.dim) or sqrt_pi.shape == (1, self.dim), "sqrt_pi must have shape [L, DIM] or [1, DIM]"
        assert S.shape[0] == sqrt_pi.shape[0], "S and sqrt_pi must have the same first dimension, either L or 1"
        assert S.dtype == self.dtype, "S must have the same dtype as the tree"
        assert sqrt_pi.dtype == self.dtype, "sqrt_pi must have the same dtype as the tree"
        return self.tree.calculate_gradients(S, sqrt_pi)

    def get_sequence_length(self) -> int:
        """Returns L, the length of the sequences"""
        return self.L
