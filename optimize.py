import sys

import torch

import input
import cat
import felsenstein_rs
import felsenstein

fasta_file = sys.argv[1]
newick_file = sys.argv[2]
rust = sys.argv[3] == 'rust'

dtype = torch.float64

tree, L = input.read_newick_fasta(newick_file, fasta_file)

torch.manual_seed(0)

#Init random parameters
shared = torch.rand(190, requires_grad=True)
energies = torch.rand(L, 20, requires_grad=True)

S, sqrt_pi = cat.rate_matrix(shared, energies)

tree = [(par, dist, input.amino_to_embedding(seq)) for par, dist, seq in tree]

if rust:
    leaf_log_p = torch.array([seq for _,_, seq in tree if seq is not None], dtype=dtype).transpose(1,0,2)
    tree = [(par, dist) for par, dist, _ in tree]
    tree = felsenstein_rs.FTreeDouble(20, tree, leaf_log_p, 1e-4)
    
else:
    tree = felsenstein.FelsensteinTree(tree)