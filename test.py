from input import read_tree_file
import felsenstein


def test_input():
    tree = read_tree_file('tree.tree')
    felstree = felsenstein.FelsensteinTree(tree)