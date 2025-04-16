import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from . import felsenstein_tree
import numpy as np

def rate_prior(S, sqrt_pi):
    S = jnp.triu(S)
    S = S + S.transpose()
    
    rate = jnp.diag(1/sqrt_pi) @ S @ jnp.diag(sqrt_pi)
    rate = rate - jnp.diag(jnp.sum(rate, axis=1))
    prior = jnp.log(sqrt_pi) * 2

    return rate, prior
    

class FelsensteinTree:
    def __init__(self, tree, leaf_log_p, distance_threshold=1e-4):
        self.tree = felsenstein_tree.FelsensteinTree(tree, distance_threshold)
        self.leaf_log_p = jnp.array(leaf_log_p)
        
    def calculate_gradients_jax(self, S, sqrt_pi, leaf_log_p) -> dict:
        
        (rate, prior), rate_prior_vjp = jax.vjp(rate_prior, S, sqrt_pi)
        
        # Calculate the gradients using the Felsenstein algorithm
        log_p, all_log_p = self.tree.log_p(rate, prior, leaf_log_p)
        
        # Calculate the gradients
        rate_grad, prior_grad = self.tree.gradients(rate, prior, all_log_p)
        grad_s, grad_sqrt_pi = rate_prior_vjp((rate_grad, prior_grad))
        
        return {"grad_s": grad_s, "grad_sqrt_pi": grad_sqrt_pi, "log_likelihood": log_p}
    
    def calculate_gradients(self, S, sqrt_pi) -> dict:
        vmap_grad = jax.vmap(self.calculate_gradients_jax, in_axes=(0, 0, 0))
        
        result = vmap_grad(S, sqrt_pi, self.leaf_log_p)
        return {"grad_s": np.array(result['grad_s']), 
                "grad_sqrt_pi": np.array(result['grad_sqrt_pi']),
                "log_likelihood": np.array(result['log_likelihood'])}