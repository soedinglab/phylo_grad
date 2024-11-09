from . import _phylo_grad
import numpy as np

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
        
    def calculate_gradients(self, S, sqrt_pi):
        return self.tree.calculate_gradients(S, sqrt_pi)