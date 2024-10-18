import torch

from input import read_tree_file
import felsenstein

import felsenstein_rs

import numpy as np


torch.manual_seed(0)


def leaf_log_p(tree):
    a = np.array([seq for _,_, seq in tree if seq is not None], dtype=np.float64)
    print(a.shape)
    return a.transpose(1,0,2)

def test_input():
    tree = read_tree_file('tree.tree')
    felstree = felsenstein.FelsensteinTree(tree)
    
def test_liklelihood():
    tree = read_tree_file('tree.tree')
    torch_tree = felsenstein.FelsensteinTree(tree)
    S = torch.exp(torch.randn(119,4, 4, dtype=torch.float64))
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(torch.randn(119,4, dtype=torch.float64), dim = 1))
    torch_logP = torch_tree.log_likelihood(S, sqrt_pi)
    
    rust_tree = felsenstein_rs.FTree('tree.tree', 1e-4)
    llp = leaf_log_p(tree)
    print(llp.shape)
    result = rust_tree.infer_param_unpaired(llp, S.numpy(), sqrt_pi.numpy())
    
    print(result['log_likelihood'])
    print(torch_logP)
    
    assert(np.allclose(result['log_likelihood'], torch_logP.numpy(), atol=1e-4))
    

def test_grads():
    tree = read_tree_file('tree.tree')
    torch_tree = felsenstein.FelsensteinTree(tree)
    S = torch.exp(torch.randn(119,4, 4, dtype=torch.float64))
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(torch.randn(119,4, dtype=torch.float64), dim = 1))
    torch_S_grad, torch_sqrt_pi_grad = felsenstein.gradients(torch_tree, S, sqrt_pi)
    
    rust_tree = felsenstein_rs.FTree('tree.tree', 1e-4)
    llp = leaf_log_p(tree)
    result = rust_tree.infer_param_unpaired(llp, S.detach().numpy(), sqrt_pi.detach().numpy())
    
    print(torch_sqrt_pi_grad)
    print(result['grad_sqrt_pi'])
    
    assert(np.allclose(result['grad_sqrt_pi'], torch_sqrt_pi_grad, atol=1e-4))
    assert(np.allclose(result['grad_delta'], torch_S_grad, atol=1e-4))