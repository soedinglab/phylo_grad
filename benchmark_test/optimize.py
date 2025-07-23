"""
    This script is the benchmark, it supports the phylo_grad gradients and pytorch gradients.
"""

import sys
import time, itertools
import resource
import torch

import input
import cat
import felsenstein
import numpy as np
import argparse

# Command line parsing, this can be mostly ignored

parser = argparse.ArgumentParser(prog='optimize_benchmark')

group = parser.add_argument_group('Leaf data', 'Alginment or distribution of leaf data')
exclusive_group = group.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--fasta_amino', help='Fasta file with amino acid sequences')

parser.add_argument('--newick', help='Newick file with tree structure')

backend = parser.add_argument_group('Backend')
exclusive_group = backend.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--rust', action='store_true')
exclusive_group.add_argument('--pytorch', action='store_true')
exclusive_group.add_argument('--jax_gpu', action='store_true')
exclusive_group.add_argument('--pytorch_gpu', action='store_true')

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
    if args.pytorch_gpu:
        tree = [(par, dist, input.amino_to_embedding(seq).cuda() if seq else None) for par, dist, seq in tree]
    else:
        tree = [(par, dist, input.amino_to_embedding(seq)) for par, dist, seq in tree]


#Init random parameters
torch.manual_seed(0)

if args.pytorch_gpu:
    shared = torch.rand(190, requires_grad=True, dtype=dtype, device="cuda")
    energies = torch.rand(L, 20, requires_grad=True, dtype=dtype, device="cuda")
else:
    shared = torch.rand(190, requires_grad=True, dtype=dtype)
    energies = torch.rand(L, 20, requires_grad=True, dtype=dtype)

if args.rust or args.jax_gpu:
    import phylo_grad
    leaf_log_p = torch.stack([seq for _,_, seq in tree if seq is not None]).transpose(1,0)
    tree = np.array([(par, dist) for par, dist, _ in tree], dtype=np_dtype)
    tree = phylo_grad.FelsensteinTree(tree, leaf_log_p.type(dtype).numpy(), 1e-4, gpu = args.jax_gpu)
    
else:
    tree = felsenstein.FelsensteinTree(tree)
    
optimizer = torch.optim.Adam([shared, energies], lr=0.01)


start = time.time()

for i in itertools.count():
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi = cat.rate_matrix(shared, energies)
    
    if args.rust or args.jax_gpu:
        result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
        S.backward(-torch.tensor(result['grad_s'], dtype=dtype))
        sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
        #print(result['log_likelihood'].sum())
        
    else:
        log_p = tree.log_likelihood(S, sqrt_pi).sum()
        loss = -log_p
        #print(loss.item())
        loss.backward()
    optimizer.step()
    
    if time.time() - start > 20 * 60:
        print(i+1)
        # Print peak memory usage
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        
        break