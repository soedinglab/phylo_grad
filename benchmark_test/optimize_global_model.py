import sys

import torch

import input
import phylo_grad
import numpy as np
import argparse

# Command line parsing, this can be mostly ignored

parser = argparse.ArgumentParser(prog='optimize_benchmark')

parser.add_argument('--fasta_amino', help='Fasta file with amino acid sequences')

parser.add_argument('--newick', help='Newick file with tree structure')

parser.add_argument('--output', help='Output file with parameters')

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
    

alignment = input.read_fasta_numeric(args.fasta_amino)
# Counts amino acids
counts = torch.zeros(21, dtype=torch.int64)
for seq in alignment.values():
    L = len(seq)
    for i in seq:
        counts[i] += 1

initial_energies = torch.log(counts[:-1])
    
#Init random parameters
torch.manual_seed(0)

shared = torch.zeros(190, requires_grad=True, dtype=dtype)
energies = initial_energies.clone().to( dtype=dtype).requires_grad_(True)


leaf_log_p = input.read_fasta(args.fasta_amino)
tree = phylo_grad.FelsensteinTree.from_newick(args.newick, leaf_log_p, np_dtype, 1e-4, gpu = False)
    
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
    
    
    result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
    S.backward(-torch.tensor(result['grad_s'], dtype=dtype), retain_graph=True)
    sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
    loss = -result['log_likelihood'].sum()
    
    print(loss.item())
    if last_loss - loss.item() < 1e-1:
        print("Loss did not decrease by more than 0.1, stopping optimization")
        break
    last_loss = loss.item()
        
    optimizer.step()

S, sqrt_pi = rate_matrix(shared, energies, L)

np.savez(args.output, log_p = -last_loss ,S=S.detach().numpy()[0], sqrt_pi=sqrt_pi.detach().numpy()[0], shared=shared.detach().numpy(), energies=energies.detach().numpy())

# Print peak memory usage
import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)