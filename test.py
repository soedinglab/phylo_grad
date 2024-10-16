import torch

from input import read_tree_file
import felsenstein


def test_input():
    tree = read_tree_file('tree.tree')
    felstree = felsenstein.FelsensteinTree(tree)
    
def test_liklelihood():
    tree = read_tree_file('tree.tree')
    felstree = felsenstein.FelsensteinTree(tree)
    S = torch.exp(torch.randn(119,5, 5))
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(torch.randn(119,5), dim = 1))
    print(felstree.log_likelihood(S, sqrt_pi))