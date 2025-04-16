import jax
import jax.numpy as jnp

class FelsensteinNode:
    def __init__(self, idx: int) -> None:
        self.children = []
        self.idx = idx

    def get_computation_plan(self) -> list[tuple[int, int]]:
        """
        Returns a computation plan for the Felsenstein algorithm.
        The plan is a list of tuples, where each tuple (child, parent) means that the log_p_transformation of child will be added to the log_p of parent
        """
        plan = []
        for child in self.children:
            plan += child.get_computation_plan()
            plan.append((child.idx, self.idx))
        
        return plan

def log_p_transformation(t : float, rate_matrix : jnp.ndarray, log_p : jnp.ndarray) -> jnp.ndarray:
    matrix_exp = jax.scipy.linalg.expm(t * rate_matrix)
    log_trans = jnp.log(matrix_exp)
    new_log_p = jax.scipy.special.logsumexp(jnp.expand_dims(log_p, 0) + log_trans, axis=1)
    return new_log_p

class FelsensteinTree:
    def __init__(self, parent_list : jnp.ndarray, distance_threshold : float = 1e-4) -> None:
        
        self.nodes = [FelsensteinNode(idx) for idx in range(len(parent_list))]
        
        self.root = None
        
        branch_lengths = []
        for child_idx, (parent_idx, branch_length) in enumerate(parent_list):
            if parent_idx == -1:
                self.root = child_idx
            else:
                self.nodes[int(parent_idx)].children.append(self.nodes[child_idx])
            branch_lengths.append(max(branch_length, distance_threshold))
        
        assert(self.root is not None), "Tree has no root"
        
        self.branch_lengths = jnp.array(branch_lengths)
        plan = self.nodes[self.root].get_computation_plan()
        self.computation_plan = jnp.array(plan, dtype=jnp.int32)     
        
    def log_p(self, rate_matrix : jnp.ndarray, prior: jnp.ndarray, leaf_log_p) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        internal_nodes = len(self.nodes) - leaf_log_p.shape[-2]
    
        # Extent the leaf_log_p to include the internal nodes
        global_log_p = jnp.pad(leaf_log_p, ((0, internal_nodes), (0, 0)), mode='constant', constant_values=0.0)
        
        def do_step(idx, current_global_log_p):
            child, parent = self.computation_plan[idx]
            branch_length = self.branch_lengths[child]
            new_log_p = log_p_transformation(branch_length, rate_matrix, current_global_log_p[child])
            return current_global_log_p.at[parent].add(new_log_p)
        # Loop over the computation plan
        log_p_all = jax.lax.fori_loop(0, self.computation_plan.shape[0], do_step, global_log_p)
        
        log_p_root = log_p_all[self.root]
        # Add the prior
        return jax.scipy.special.logsumexp(log_p_root + prior, axis=0), log_p_all
    
    def gradients(self, rate_matrix : jnp.ndarray, prior: jnp.ndarray, global_log_p) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the gradients of the log likelihood with respect to the rate matrix and the prior.
        """
        
        # We to backward diff, so we need to reverse the computation plan
        reverse_computation_plan = jnp.flip(self.computation_plan, axis=0)
        
        def do_step(idx, data):
            current_log_p_grad, current_grad_rate_matrix = data
            child, parent = reverse_computation_plan[idx]
            branch_length = self.branch_lengths[child]
            _, vjp = jax.vjp(log_p_transformation, branch_length, rate_matrix, global_log_p[child])
            _, grad_rate_matrix, grad_log_p = vjp(current_log_p_grad[parent])
            
            return current_log_p_grad.at[child].set(grad_log_p), current_grad_rate_matrix + grad_rate_matrix
        
        root_log_p_grad, prior_grad = jax.grad(lambda log_p, prior: jax.scipy.special.logsumexp(log_p + prior, axis=0), argnums=[0,1])(global_log_p[self.root], prior)
        
        log_p_grad = jnp.zeros_like(global_log_p)
        log_p_grad = log_p_grad.at[self.root].set(root_log_p_grad)
        _, grad_rate_matrix = jax.lax.fori_loop(0, reverse_computation_plan.shape[0], do_step, (log_p_grad, jnp.zeros_like(rate_matrix)))
        return grad_rate_matrix, prior_grad
        