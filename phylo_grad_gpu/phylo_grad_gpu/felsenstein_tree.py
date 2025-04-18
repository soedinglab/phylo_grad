import jax
import jax.numpy as jnp

MIN_SQRT_PI = 1e-10

def precomp_S_pi(S : jnp.ndarray, sqrt_pi : jnp.ndarray) -> dict:

    S = jnp.triu(S, k=1)
    S = S + S.T
    sqrt_pi = jnp.clip(sqrt_pi, min=MIN_SQRT_PI)
    sqrt_pi_inv = 1.0 / sqrt_pi
    sqrt_pi = jnp.diag(sqrt_pi)
    sqrt_pi_inv = jnp.diag(sqrt_pi_inv)

    Q = sqrt_pi_inv @ S @ sqrt_pi

    S = S - jnp.diag(Q.sum(axis=1))

    eigenvalues, eigenvectors = jnp.linalg.eigh(S)

    B = sqrt_pi_inv @ eigenvectors
    B_inv = eigenvectors.T @ sqrt_pi

    return {'B': B, 'B_inv': B_inv, 'eigenvalues': eigenvalues}


def X(t : float, eigenvalues: jnp.ndarray) -> jnp.ndarray:

    exp = jnp.exp(t * eigenvalues)

    exp_i = jnp.expand_dims(exp, axis = 1) # (i, j) of the array will be i
    exp_j = jnp.expand_dims(exp, axis = 0) # (i, j) of the array will be j

    eigen_i = jnp.expand_dims(eigenvalues, axis = 1) # (i, j) of the array will be i
    eigen_j = jnp.expand_dims(eigenvalues, axis = 0) # (i, j) of the array will be j

    small_diff = exp_i

    big_diff = (exp_i - exp_j) / (eigen_i - eigen_j)

    medium_diff = exp_j * jnp.expm1(t * (eigen_i - eigen_j)) / (eigen_i - eigen_j)

    diff = jnp.abs(eigen_i - eigen_j)

    big = diff > 10
    small = diff < 1e-10
    medium = jnp.logical_not(big) & jnp.logical_not(small)

    big_diff = jnp.where(big, big_diff, 0)
    small_diff = jnp.where(small, small_diff, 0)
    medium_diff = jnp.where(medium, medium_diff, 0)
    X = jnp.where(big, big_diff, 0) + jnp.where(small, small_diff, 0) + jnp.where(medium, medium_diff, 0)

    return X

@jax.custom_gradient
def custom_matrix_exp(t : float, precomp_S_pi : dict) -> jnp.ndarray:
    B = precomp_S_pi['B']
    B_inv = precomp_S_pi['B_inv']
    eigenvalues = precomp_S_pi['eigenvalues']

    # Compute the matrix exponential
    exp_eigenvalues = jnp.exp(t * eigenvalues)
    exp_matrix = B @ jnp.diag(exp_eigenvalues) @ B_inv

    def grad(cotangent):
        tmp = B.T @ cotangent @ B_inv.T

        tmp2 = tmp * X(t, eigenvalues)

        # We use B for the gradient of Q, the other gradients are 0
        return 0.0, {'B' : B_inv.T @ tmp2 @ B.T, 
                     'B_inv' : 0.0, 
                     'eigenvalues' : 0.0}

    return exp_matrix, grad 



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

def log_p_transformation(t : float, precomp_S_pi : dict, log_p : jnp.ndarray) -> jnp.ndarray:
    matrix_exp = custom_matrix_exp(t, precomp_S_pi)
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
        