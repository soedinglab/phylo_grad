"""
    This script is the benchmark, it supports the phylo_grad gradients and pytorch gradients.
"""

import sys

import torch

import input
import phylo_grad
import felsenstein
import numpy as np
import argparse

# Command line parsing, this can be mostly ignored

parser = argparse.ArgumentParser(prog='optimize_benchmark')

group = parser.add_argument_group('Leaf data', 'Alginment or distribution of leaf data')
exclusive_group = group.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--fasta_amino', help='Fasta file with amino acid sequences')

parser.add_argument('--newick', help='Newick file with tree structure')

parser.add_argument('--output', help='Output file with parameters')

backend = parser.add_argument_group('Backend')
exclusive_group = backend.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument('--rust', action='store_true')
exclusive_group.add_argument('--pytorch', action='store_true')
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
    # Counts amino acids
    counts = torch.zeros(21, dtype=torch.int64)
    for _, _, seq in tree:
        if seq is not None:
            numeric = [input.amino_mapping[c] for c in seq]
            for i in numeric:
                counts[i] += 1
    
    initial_energies = torch.log(counts[:-1])

    if args.pytorch_gpu:
        tree = [(par, dist, input.amino_to_embedding(seq).cuda() if seq else None) for par, dist, seq in tree]
    else:
        tree = [(par, dist, input.amino_to_embedding(seq)) for par, dist, seq in tree]
    
    


#Init random parameters
torch.manual_seed(0)

if args.pytorch_gpu:
    shared = torch.zeros(190, requires_grad=True, dtype=dtype, device="cuda")
    energies = torch.tensor(initial_energies, requires_grad=True, dtype=dtype, device="cuda")
else:
    shared = torch.zeros(190, requires_grad=True, dtype=dtype)
    energies = torch.tensor(initial_energies, requires_grad=True, dtype=dtype)

if args.rust:
    leaf_log_p = torch.stack([seq for _,_, seq in tree if seq is not None]).transpose(1,0)
    tree = np.array([(par, dist) for par, dist, _ in tree], dtype=np_dtype)
    tree = phylo_grad.FelsensteinTree(tree, leaf_log_p.type(dtype).numpy(), 1e-4)
    
else:
    tree = felsenstein.FelsensteinTree(tree)
    
optimizer = torch.optim.Adam([shared, energies], lr=0.1)

def rate_matrix(shared, energies, L):
    """
        shared is a [190] tensor representing the upper diagonal of the log(S) matrix
        energies is a [20] tensor representing the energy of each amino acid at each side. The distribution is given by the softmax of the energies
    """
    dtype = shared.dtype
    S = torch.zeros((20,20), dtype=dtype, device=shared.device)
    S[*torch.triu_indices(20,20, offset= 1)] = shared
    S = S + S.transpose(0,1)
    S = torch.exp(S)
    
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(energies, dim = 0))
    
    Q = torch.diag(1/sqrt_pi) @ S @ torch.diag(sqrt_pi)
    Q.fill_diagonal_(0)
    exp_mutations = sqrt_pi ** 2 * Q.sum(dim=1)
    
    S = S / exp_mutations.sum()
    
    return S.unsqueeze(0), sqrt_pi.unsqueeze(0)

last_loss = float('inf')
while True:
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi = rate_matrix(shared, energies, L)
    
    if args.rust:
        result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
        S.backward(-torch.tensor(result['grad_s'], dtype=dtype), retain_graph=True)
        sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
        loss = -result['log_likelihood'].sum()
        
    else:
        log_p = tree.log_likelihood(S, sqrt_pi).sum()
        loss = -log_p
        #print(loss.item())
        loss.backward()
    
    print(loss.item())
    if last_loss - loss.item() < 1e-1:
        print("Loss did not decrease by more than 0.1, stopping optimization")
        break
    last_loss = loss.item()
        
    optimizer.step()

S, sqrt_pi = rate_matrix(shared, energies, L)

np.savez(args.output, S=S.detach().cpu().numpy()[0], sqrt_pi=sqrt_pi.detach().cpu().numpy()[0], shared=shared.detach().cpu().numpy(), energies=energies.detach().cpu().numpy())

# Print peak memory usage
import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)