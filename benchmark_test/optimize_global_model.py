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

args = parser.parse_args(sys.argv[1:])

dtype = torch.float64
np_dtype = np.float64

alignment = input.read_fasta_numeric(args.fasta_amino)
# Counts amino acids
counts = torch.zeros(21, dtype=torch.int64)
for seq in alignment.values():
    L = len(seq)
    for i in seq:
        counts[i] += 1

initial_energies = torch.log(counts[:-1])

shared = torch.zeros(190, requires_grad=True, dtype=dtype)
energies = initial_energies.clone().to( dtype=dtype).requires_grad_(True)


leaf_log_p = input.read_fasta(args.fasta_amino)
tree = phylo_grad.FelsensteinTree.from_newick(args.newick, leaf_log_p, np_dtype, 1e-4, gpu = False)
    
optimizer = torch.optim.LBFGS([shared, energies], tolerance_change=0.01, lr=0.1)


def rate_matrix2(log_R : torch.Tensor, log_pi : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    log_pi = log_pi - torch.logsumexp(log_pi, dim=0)

    log_pi.retain_grad()

    log_R_mat = torch.zeros((20,20), dtype=log_R.dtype, device=log_R.device)
    log_R_mat[*torch.tril_indices(20,20, offset=-1)] = log_R
    log_R_mat = log_R_mat + log_R_mat.transpose(0,1)
    torch.diagonal(log_R_mat).fill_(float('-inf'))
    log_R_mat.retain_grad()


    piRpi = log_pi.unsqueeze(0) + log_R_mat + log_pi.unsqueeze(1)
    piRpi.retain_grad()

    logM = torch.logsumexp(piRpi, dim=(0,1))
    logM.retain_grad()

    logS = 0.5 * (log_pi.unsqueeze(0) + log_pi.unsqueeze(1)) + log_R_mat - logM
    logS.retain_grad()

    return torch.exp(logS).unsqueeze(0), torch.exp(0.5 * log_pi).unsqueeze(0), logM, logS, piRpi, log_R_mat, log_pi

def rate_matrix_backward(log_R : torch.Tensor, log_pi_orig : torch.Tensor, tangent_S, tangent_sqrt_pi) -> tuple[torch.Tensor, torch.Tensor]:
    log_pi = log_pi_orig - torch.logsumexp(log_pi_orig, dim=0)

    log_R_mat = torch.zeros((20,20), dtype=log_R.dtype, device=log_R.device)
    log_R_mat[*torch.tril_indices(20,20, offset=-1)] = log_R
    log_R_mat = log_R_mat + log_R_mat.transpose(0,1)
    torch.diagonal(log_R_mat).fill_(float('-inf'))


    piRpi = log_pi.unsqueeze(0) + log_R_mat + log_pi.unsqueeze(1)

    logM = torch.logsumexp(piRpi, dim=(0,1))

    logS = 0.5 * (log_pi.unsqueeze(0) + log_pi.unsqueeze(1)) + log_R_mat - logM


    d_logS = tangent_S * torch.exp(logS)

    torch.diagonal(d_logS).fill_(0.0)

    d_logM = - d_logS.sum()

    d_piRpi = torch.softmax(piRpi.flatten(), dim=0).reshape(20, 20) * d_logM

    d_log_R_mat = d_logS + d_piRpi

    d_log_R = d_log_R_mat[*torch.tril_indices(20,20, offset=-1)] + d_log_R_mat.transpose(0,1)[*torch.tril_indices(20,20, offset=-1)]

    d_log_pi = (d_logS * 0.5).sum(dim=0) + (d_logS * 0.5).sum(dim=1) + d_piRpi.sum(dim=0) + d_piRpi.sum(dim=1)

    d_log_pi = d_log_pi + tangent_sqrt_pi * 0.5 * (torch.exp(0.5 * log_pi))

    d_log_pi_orig = d_log_pi - torch.softmax(log_pi_orig, dim=0) * d_log_pi.sum()

    return d_log_R, d_log_pi_orig

def rate_matrix(log_R : torch.Tensor, log_pi : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
        S and sqrt_pi for the GTR20 model like in IQ-TREE
        log_R is a vector with lower diagonal of log rate matrix R of shape (190)
        log_pi is a vector of shape (20)
    """

    pi = torch.softmax(log_pi, dim = 0)
    R = torch.zeros((20,20), dtype=log_R.dtype, device=log_R.device)
    R[*torch.tril_indices(20,20, offset=-1)] = torch.exp(log_R)
    R = R + R.transpose(0,1)
    Q = R @ torch.diag_embed(pi)
    
    Q = Q - torch.diag(Q.sum(dim=1))
    
    sqrt_pi = torch.sqrt(pi)

    Q = Q / - (Q.diag() * pi).sum()

    S = torch.diag(sqrt_pi) @ Q @ torch.diag(1/sqrt_pi)
    return S.unsqueeze(0), sqrt_pi.unsqueeze(0)


last_loss = float('inf')

def closure():
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi= rate_matrix(shared, energies)

    
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

final_result = tree.calculate_log_likelihoods(S.detach().numpy(), sqrt_pi.detach().numpy())
print(f"BEST SCORE FOUND : {final_result.sum()}")


S = S[0]
sqrt_pi = sqrt_pi[0]

# Save the parameters as PAML file

with open(args.output, 'w') as f:
    R = torch.diag(1/sqrt_pi) @ S @ torch.diag(1/sqrt_pi)

    row, col = torch.tril_indices(20,20, offset=-1)

    entries = R[row, col].cpu().detach().numpy()

    for e in entries:
        f.write(f"{e} ")
    f.write("\n")

    for pi in (sqrt_pi ** 2).cpu().detach().numpy():
        f.write(f"{pi} ")

end = timeit.default_timer()
print(f"Total wall-clock time used: {end - start} sec")