import io

from . import _phylo_grad
import numpy as np
from Bio import Phylo

def process_newick(newick: str | io.TextIOBase, leaf_pl_dict : dict) -> dict:
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
    branch_lengths = []
    leaf_log_p = []
    
    for node in nodes:
        parent_list.append(node.parent_id)
        branch_lengths.append(node.branch_length)
        if node.is_terminal():
            if node.name not in leaf_pl_dict:
                raise ValueError(f"Leaf {node.name} from newick not found in leaf_pl_dict")
            leaf_log_p.append(leaf_pl_dict[node.name])
    
    leaf_pl_array = np.array(leaf_log_p, np.float64).transpose(1, 0, 2)

    return {
        'parent_list': np.array(parent_list, dtype=np.int32),
        'branch_lengths': np.array(branch_lengths, dtype=np.float64),
        'leaf_pl': leaf_pl_array
    }

class FelsensteinTree:
    """A class to represent a phylogenetic tree, it contains the tree topology and edge lengths as well as the log probabilities of the leaf nodes
       It's easiest to use the from_newick method to create a FelsensteinTree object.
    """
    def __init__(self, parent_list, branch_lengths, leaf_pl, distance_threshold=1e-4):
        """Typically you should prefer to use the from_newick method to create a FelsensteinTree object.
           For more information see the Rust documentation of phylo_grad
        """
        assert isinstance(parent_list, np.ndarray)
        assert isinstance(branch_lengths, np.ndarray)
        assert isinstance(leaf_pl, np.ndarray)
        assert isinstance(distance_threshold, float)
        
        assert len(leaf_pl.shape) == 3, "leaf_pl must have shape [L, N, DIM]"
        
        dim = leaf_pl.shape[2]

        self.dtype = leaf_pl.dtype
        self.L = leaf_pl.shape[0]
        self.dim = dim

        branch_lengths = np.clip(branch_lengths, distance_threshold, None)
        
        try:
            tree_class = getattr(_phylo_grad, f'Backend_{dim}')
        except AttributeError:
            raise ValueError(f'Unsupported dim {dim}, see Readme.md for more information')

        self.tree = tree_class(parent_list.astype(np.int32), branch_lengths.astype(np.float64))
        self.tree.bind_leaf_pl(leaf_pl.astype(np.float64))

    @classmethod
    def from_newick(cls, newick : str | io.TextIOBase, leaf_pl_dict : dict, distance_threshold : float = 1e-4) -> 'FelsensteinTree':
        """
            Preferred method to create a FelsensteinTree object.
            
            Each of the leaf nodes in the tree needs an array of shape [L, DIM] where L is the sequence length and DIM is number of states (20 for amino acids, 4 for nucleotides, other custom states).
            The entries represent the probabilities of the states at that position.
            For example with the 4 states A=0, C=1, G=2, T=3, the probabilities array of the sequence "ACGT" would be:
            [[1.0, 0.0, 0.0, 0.0], # A
            [0.0, 1.0, 0.0, 0.0], # C
            [0.0, 0.0, 1.0, 0.0], # G
            [0.0, 0.0, 0.0, 1.0]] # T
            A gap can be represented by [1.0,1.0,1.0,1.0] meaning all possible states have a probability of 1.0 resulting in ignoring the gap in the likelihood calculation.
            
            :param str newick: File handle or File name containing a tree in newick format
            :param dict leaf_pl_dict: A dictionary mapping leaf names to probabilities, {seq_id : p_array} where p_array has shape [L, DIM] and contains the probabilities of the states at each position for that leaf node. L is the sequence length and DIM is number of states (20 for amino acids, 4 for nucleotides, other custom states).
            :param float distance_threshold: Every edge of the tree will set to be at least that long because very small edge lengths can lead to numerical unstabilities.
        """

        data = process_newick(newick, leaf_pl_dict)
        return cls(data['parent_list'], data['branch_lengths'], data['leaf_pl'], distance_threshold) 
        
    def calculate_gradients(self, S : np.ndarray, sqrt_pi : np.ndarray) -> dict:
        """Calculates the gradients of the log likelihood with respect to the given substitution model.
        
        Details of the parametrization of the substitution model can be found in the PhyloGrad paper.
        
        In case one of the symmetric matrices are not diagonalizable, the gradients will be 0.0 and the log likelihood will be -inf.
        This only happens when the matrix has very big or very small eigenvalues (absolute value bigger than 1e5).

        For S only the upper triangle is used, the lower triangle is ignored in the input and for the gradient output it will be filled with zeros.

        :param S: The symmetric S matrix of the substitution model, shape [L, DIM, DIM] or [1, DIM, DIM] for broadcasting which will be faster but currently uses more memory
        :param sqrt_pi: The square root of the stationary distribution, shape [L, DIM] or [1, DIM] for broadcasting which will be faster but currently uses more memory
        
        :return: dict with the keys 'grad_s', 'grad_sqrt_pi' and 'log_likelihood'
                 'grad_s' has the same shape as S, 'grad_sqrt_pi' has the same shape as sqrt_pi and are the gradients
                 'log_likelihood' has shape [L] and is the log likelihood of each column given the substitution model
                 
        """
        assert isinstance(S, np.ndarray), "S must be a numpy array"
        assert S.shape == (self.L, self.dim, self.dim) or S.shape == (1, self.dim, self.dim), "S must have shape [L, DIM, DIM] or [1, DIM, DIM]"
        assert isinstance(sqrt_pi, np.ndarray), "sqrt_pi must be a numpy array"
        assert sqrt_pi.shape == (self.L, self.dim) or sqrt_pi.shape == (1, self.dim), "sqrt_pi must have shape [L, DIM] or [1, DIM]"
        assert S.shape[0] == sqrt_pi.shape[0], "S and sqrt_pi must have the same first dimension, either L or 1"
        return self.tree.calculate_gradients(S.astype(np.float64), sqrt_pi.astype(np.float64))
    
    def calculate_log_likelihoods(self, S : np.ndarray, sqrt_pi : np.ndarray) -> np.ndarray:
        """
            Same as calculate_gradients but only returns the log likelihoods and does not compute gradients.
        """
        assert isinstance(S, np.ndarray), "S must be a numpy array"
        assert S.shape == (self.L, self.dim, self.dim) or S.shape == (1, self.dim, self.dim), "S must have shape [L, DIM, DIM] or [1, DIM, DIM]"
        assert isinstance(sqrt_pi, np.ndarray), "sqrt_pi must be a numpy array"
        assert sqrt_pi.shape == (self.L, self.dim) or sqrt_pi.shape == (1, self.dim), "sqrt_pi must have shape [L, DIM] or [1, DIM]"
        assert S.shape[0] == sqrt_pi.shape[0], "S and sqrt_pi must have the same first dimension, either L or 1"
        return self.tree.calculate_log_likelihoods(S.astype(np.float64), sqrt_pi.astype(np.float64))

    def get_sequence_length(self) -> int:
        """Returns L, the length of the sequences"""
        return self.L
