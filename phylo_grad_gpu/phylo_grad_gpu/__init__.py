import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from . import felsenstein_tree
import numpy as np


class FelsensteinTree:
    def __init__(self, parent_list, branch_lengths, leaf_log_p):
        self.tree = felsenstein_tree.FelsensteinTree(jnp.array(parent_list), jnp.array(branch_lengths))
        self.leaf_log_p = jnp.array(leaf_log_p)
        self.vmap_grad = jax.jit(jax.vmap(self.calculate_gradients_jax, in_axes=(0, 0, 0)))
        
    def calculate_gradients_jax(self, S, sqrt_pi, leaf_log_p) -> dict:
        
        # Forward pass to calculate log likelihood
        log_p, all_log_p, precomp = self.tree.log_p(S, sqrt_pi, leaf_log_p)
        
        # Backward pass to calculate gradients
        grad_s, grad_sqrt_pi = self.tree.gradients(S, sqrt_pi, all_log_p, precomp)
        
        return {"grad_s": grad_s, "grad_sqrt_pi": grad_sqrt_pi, "log_likelihood": log_p}
    
    def calculate_gradients(self, S, sqrt_pi) -> dict:
        result = self.vmap_grad(S, sqrt_pi, self.leaf_log_p)
        return {"grad_s": np.array(result['grad_s']), 
                "grad_sqrt_pi": np.array(result['grad_sqrt_pi']),
                "log_likelihood": np.array(result['log_likelihood'])}