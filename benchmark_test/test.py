import itertools

import torch

from input import read_newick
import felsenstein

import phylo_grad

import numpy as np


def gen_tree(t_dtype, dim):
    torch.manual_seed(0)
    newick = read_newick('test_tree.newick')
    L = 300
    leaf_log_p = torch.randn([L, newick['num_leaf'], dim], dtype=t_dtype)
    leaf_log_p = torch.nn.functional.log_softmax(leaf_log_p, dim=2)
    
    pytroch_tree = [(par, dist, llp.double() if llp is not None else None) for par, dist, llp in itertools.zip_longest(newick["parent_list"], newick["branch_lengths"], leaf_log_p.transpose(0, 1))]

    return felsenstein.FelsensteinTree(pytroch_tree), np.array(newick["parent_list"], dtype=np.int32), np.array(newick['branch_lengths']), leaf_log_p

def gen_data(t_dtype, dim, seed):
    torch.manual_seed(seed)
    L = 300
    S = torch.exp(torch.randn(L,dim, dim, dtype=t_dtype))
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(torch.randn(L,dim, dtype=t_dtype), dim = 1))
    return S, sqrt_pi

def gen_data_single_model(t_dtype, dim, seed):
    S, sqrt_pi = gen_data(t_dtype, dim, seed)
    S = S[0:1]
    sqrt_pi = sqrt_pi[0:1]
    return S, sqrt_pi

def helper_test(dtype, dim : int, gradients: bool, single_model: bool = False, gpu: bool = False):
    if dtype == "f32":
        t_dtype = torch.float32
        np_dtype = np.float32
    else:
        t_dtype = torch.float64
        np_dtype = np.float64
    
    if dtype == "f32":
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 1e-4
        atol = 1e-4

    torch_tree, parent_list, branch_lengths, leaf_log_p = gen_tree(t_dtype, dim)

    if single_model:
        data = [gen_data_single_model(t_dtype, dim, seed) for seed in range(10)]
    else:
        data = [gen_data(t_dtype, dim, seed) for seed in range(10)]

    rust_tree = phylo_grad.FelsensteinTree(parent_list, branch_lengths.astype(np_dtype), leaf_log_p.numpy(), 1e-4, gpu)

        
    for i in range(10):
        S, sqrt_pi = data[i]
        torch_logP = torch_tree.log_likelihood(S, sqrt_pi)


        result = rust_tree.calculate_gradients(S.numpy(), sqrt_pi.numpy())    


        assert(np.allclose(result['log_likelihood'], torch_logP.numpy(), rtol=rtol))
        
        if gradients:
            torch_S_grad, torch_sqrt_pi_grad = felsenstein.gradients(torch_tree, S.double(), sqrt_pi.double())
            
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
            
            
            assert(np.allclose(result['grad_sqrt_pi'], torch_sqrt_pi_grad, rtol=rtol, atol=atol))
            assert(np.allclose(result['grad_s'], torch_S_grad, rtol=rtol, atol=atol))
    
def test_likelihood():
    helper_test("f32", 4, False)
    helper_test("f32", 20, False)
    helper_test("f64", 4, False)
    helper_test("f64", 20, False)

    helper_test("f64", 4, False, gpu = True)
    helper_test("f64", 20, False, gpu= True)
    helper_test("f32", 4, False, gpu= True)
    helper_test("f32", 20, False, gpu= True)

def test_grads():
    helper_test("f32", 4, True)
    helper_test("f32", 20, True)
    helper_test("f64", 4, True)
    helper_test("f64", 20, True)
    
def test_grads_single_model():
    helper_test("f32", 4, True, single_model=True)
    helper_test("f32", 20, True, single_model=True)
    helper_test("f64", 4, True, single_model=True)
    helper_test("f64", 20, True, single_model=True)