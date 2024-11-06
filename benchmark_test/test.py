import itertools

import torch

from input import read_newick
import felsenstein

import phylo_grad

import numpy as np


torch.manual_seed(0)

def gen_data(t_dtype, dim):
    tree_top, num_leaf = read_newick('test_tree.newick')
    L = 300
    S = torch.exp(torch.randn(L,dim, dim, dtype=t_dtype))
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(torch.randn(L,dim, dtype=t_dtype), dim = 1))
    leaf_log_p = torch.randn([L, num_leaf, dim], dtype=t_dtype)
    leaf_log_p = torch.nn.functional.log_softmax(leaf_log_p, dim = 2)
    
    pytroch_tree = [(par, dist, llp) for (par, dist), llp in itertools.zip_longest(tree_top, leaf_log_p.transpose(0,1))]
    
    return felsenstein.FelsensteinTree(pytroch_tree), S, sqrt_pi, leaf_log_p, tree_top
    

    
def helper_test(dtype, dim : int, gradients: bool):
    if dtype == "f32":
        t_dtype = torch.float32
        np_dtype = np.float32
    else:
        t_dtype = torch.float64
        np_dtype = np.float64
    
    torch_tree, S, sqrt_pi, leaf_log_p, tree_top = gen_data(t_dtype, dim)
    
    torch_logP = torch_tree.log_likelihood(S, sqrt_pi)
    
    np_tree = np.array(tree_top, dtype=np_dtype)
    rust_tree = phylo_grad.FelsensteinTree(np_tree, leaf_log_p.numpy(), 1e-4)
    
    result = rust_tree.calculate_gradients(S.numpy(), sqrt_pi.numpy())
    
    assert(np.allclose(result['log_likelihood'], torch_logP.numpy(), rtol=1e-4))
    
    if gradients:
        torch_S_grad, torch_sqrt_pi_grad = felsenstein.gradients(torch_tree, S, sqrt_pi)
        
        if torch_S_grad.isnan().any():
            print("Torch S grad is nan")
        
        if torch_sqrt_pi_grad.isnan().any():
            print("Torch sqrt pi grad is nan")
        
        if np.isnan(result['grad_s']).any():
            print("Rust S grad is nan")
            print(np.where(np.isnan(result['grad_s'])))
        
        if np.isnan(result['grad_sqrt_pi']).any():
            print("Rust sqrt pi grad is nan")
            print(np.where(np.isnan(result['grad_sqrt_pi'])))
        
        if dtype == "f32":
            rtol = 1e-2
            atol = 1e-2
        else:
            rtol = 1e-4
            atol = 1e-4
        
        assert(np.allclose(result['grad_sqrt_pi'], torch_sqrt_pi_grad, rtol=rtol, atol=atol))
        assert(np.allclose(result['grad_s'], torch_S_grad, rtol=rtol, atol=atol))
    
def test_liklelihood():
    helper_test("f32", 4, False)
    helper_test("f64", 4, False)
    helper_test("f64", 20, False)


def test_grads():
    helper_test("f32", 4, True)
    helper_test("f32", 20, True)
    helper_test("f64", 4, True)
    helper_test("f64", 20, True)