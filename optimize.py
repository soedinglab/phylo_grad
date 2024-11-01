import sys

import torch

import input
import cat
import felsenstein_rs
import felsenstein
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='optimize_benchmark')

group = parser.add_argument_group('Leaf data', 'Alginment or distribution of leaf data')
exclusive_group = group.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--fasta_amino', help='Fasta file with amino acid sequences')
exclusive_group.add_argument('--leaf_dist_numpy', help='npz file with taxa names as key and [L,S] array of log probabilities')

parser.add_argument('--newick', help='Newick file with tree structure')

backend = parser.add_argument_group('Backend')
exclusive_group = backend.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--rust', action='store_true')
exclusive_group.add_argument('--pytorch', action='store_true')

fp_precision = parser.add_argument_group('fp precision')
exclusive_group = fp_precision.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--f32', action='store_true')
exclusive_group.add_argument('--f64', action='store_true')

args = parser.parse_args(sys.argv[1:])

if args.f64:
    dtype = torch.float64
    np_dtype = np.float64
else:
    dtype = torch.float32
    np_dtype = np.float32
    
if args.fasta_amino is not None:
    tree, L = input.read_newick_fasta(args.newick, args.fasta_amino)
    tree = [(par, dist, input.amino_to_embedding(seq)) for par, dist, seq in tree]
    
if args.leaf_dist_numpy is not None:
    raise NotImplementedError('Not implemented yet')


#Init random parameters
torch.manual_seed(0)
shared = torch.rand(190, requires_grad=True, dtype=dtype)
energies = torch.rand(L, 20, requires_grad=True, dtype=dtype)



if args.rust:
    leaf_log_p = torch.stack([seq for _,_, seq in tree if seq is not None]).transpose(1,0)
    tree = np.array([(par, dist) for par, dist, _ in tree], dtype=np_dtype)
    tree = felsenstein_rs.FTreeDouble(20, tree, leaf_log_p.numpy(), 1e-4)
    
else:
    tree = felsenstein.FelsensteinTree(tree)
    
optimizer = torch.optim.Adam([shared, energies], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    S, sqrt_pi = cat.rate_matrix(shared, energies)
    if args.rust:
        result = tree.infer_param_unpaired(S.detach().numpy(), sqrt_pi.detach().numpy())
        S.backward(-torch.tensor(result['grad_delta'], dtype=dtype))
        sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
        #print(result['log_likelihood'].sum())
        
    else:
        log_p = tree.log_likelihood(S, sqrt_pi).sum()
        loss = -log_p
        #print(loss.item())
        loss.backward()
    optimizer.step()

import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)