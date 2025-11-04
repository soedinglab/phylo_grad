import sys

import torch

import input
import phylo_grad
import numpy as np
import argparse
import timeit

start = timeit.default_timer()
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
    
optimizer = torch.optim.LBFGS([shared, energies], tolerance_change=0.01, lr=0.1)

def rate_matrix(log_R : torch.Tensor, log_pi : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
        log_R is a vector with lower diagonal of log rate matrix R of shape (190)
        log_pi is a vector of shape (20)
    """

    pi = torch.softmax(log_pi, dim = 0)
    print(pi, file=sys.stderr)
    R = torch.zeros((20,20), dtype=log_R.dtype, device=log_R.device)
    R[*torch.tril_indices(20,20, offset=-1)] = torch.exp(log_R)
    R = R + R.transpose(0,1)
    Q = R @ torch.diag_embed(pi)
    
    Q = Q - torch.diag(Q.sum(dim=1))
    
    sqrt_pi = torch.sqrt(pi)

    Q = Q / - (Q.diag() * pi).sum()

    S = torch.diag(sqrt_pi) @ Q @ torch.diag(1/sqrt_pi)
    return S.unsqueeze(0), sqrt_pi.unsqueeze(0)

def rate_matrix_old(shared, energies, L):
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
    
    # Normalize to have expected PAM=1
    Q = torch.diag(1/sqrt_pi) @ S @ torch.diag(sqrt_pi)
    Q.fill_diagonal_(0)
    exp_mutations = sqrt_pi ** 2 * Q.sum(dim=1)
    
    S = S / exp_mutations.sum()
    
    return S.unsqueeze(0), sqrt_pi.unsqueeze(0)


last_loss = float('inf')

def closure():
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi = rate_matrix(shared, energies)
    
    
    result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
    S.backward(-torch.tensor(result['grad_s'], dtype=dtype), retain_graph=True)
    sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi'], dtype=dtype))
    loss = -result['log_likelihood'].sum()
    
    return loss

for i in range(100):
    loss = optimizer.step(closure)
    print(f"Iteration {i}, log likelihood: {-loss.item()}")
    if abs(loss.item() - last_loss) < 0.01:
        break
    last_loss = loss.item()


S, sqrt_pi = rate_matrix(shared, energies)

S = S[0]
sqrt_pi = sqrt_pi[0]

print(S, sqrt_pi, file=sys.stderr)

# Save the parameters as PAML file

with open(args.output, 'w') as f:
    R = torch.diag(1/sqrt_pi) @ S @ torch.diag(1/sqrt_pi)

    print(R, file=sys.stderr)

    row, col = torch.tril_indices(20,20, offset=-1)

    entries = R[row, col].cpu().detach().numpy()

    for e in entries:
        f.write(f"{e} ")
    f.write("\n")

    for pi in (sqrt_pi ** 2).cpu().detach().numpy():
        f.write(f"{pi} ")

end = timeit.default_timer()
print(f"Total wall-clock time used: {end - start} sec")
# Print peak memory usage
import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)