import jax
import jax.numpy as jnp
from . import felsenstein_tree
import numpy as np

def batched_diag(arr):
    vectorized_diag = np.vectorize(np.diag, signature='(n)->(n,n)')
    return vectorized_diag(arr)

class FelsensteinTree:
    def __init__(self, tree, leaf_log_p, distance_threshold=1e-4):
        self.tree = felsenstein_tree.FelsensteinTree(tree, distance_threshold)
        self.leaf_log_p = jnp.array(leaf_log_p)
        
    def calculate_gradients(self, S, sqrt_pi) -> dict:
        
        v_log_p = jax.vmap(self.tree.log_p, in_axes=0)
        
        S = np.triu(S)
        S = S + S.swapaxes(1,2)
        
        rate = batched_diag(1/sqrt_pi) @ S @ batched_diag(sqrt_pi)
        rate = rate - batched_diag(np.sum(rate, axis=2))
        prior = np.log(sqrt_pi) * 2
        # Calculate the gradients using the Felsenstein algorithm
        log_p = v_log_p(jnp.array(rate), jnp.array(prior), self.leaf_log_p)
        return {"grad_s": None, "grad_sqrt_pi": None, "log_likelihood": np.array(log_p)}