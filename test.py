import torch

from input import read_tree_file
import felsenstein

import felsenstein_rs

import numpy as np


torch.manual_seed(0)

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
    idx = np.array(range(119), dtype=np.uint64)
    result = rust_tree.infer_param_unpaired(idx, S.numpy(), sqrt_pi.numpy(), False)
    
    print(result['log_likelihood'])
    print(torch_logP)
    
    assert(np.allclose(result['log_likelihood'], torch_logP.numpy(), atol=1e-4))