import sys

import torch

import input
import cat
import felsenstein_rs
import felsenstein
import numpy as np

fasta_file = sys.argv[1]
newick_file = sys.argv[2]
rust = sys.argv[3] == 'rust'

dtype = torch.float64

tree, L = input.read_newick_fasta(newick_file, fasta_file)

torch.manual_seed(0)

#Init random parameters
shared = torch.rand(190, requires_grad=True, dtype=dtype)
energies = torch.rand(L, 20, requires_grad=True, dtype=dtype)


tree = [(par, dist, input.amino_to_embedding(seq)) for par, dist, seq in tree]

if rust:
    leaf_log_p = torch.stack([seq for _,_, seq in tree if seq is not None]).transpose(1,0)
    tree = np.array([(par, dist) for par, dist, _ in tree], dtype=np.float64)
    tree = felsenstein_rs.FTreeDouble(20, tree, leaf_log_p.numpy(), 1e-4)
    
else:
    tree = felsenstein.FelsensteinTree(tree)
    
optimizer = torch.optim.Adam([shared, energies], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    S, sqrt_pi = cat.rate_matrix(shared, energies)
    if rust:
        result = tree.infer_param_unpaired(S.detach().numpy(), sqrt_pi.detach().numpy())
        S.backward(-torch.tensor(result['grad_delta'], dtype=dtype))
        sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
        print(result['log_likelihood'].sum())
        
    else:
        log_p = tree.log_likelihood(S, sqrt_pi).sum()
        loss = -log_p
        print(loss.item())
        loss.backward()
    optimizer.step()