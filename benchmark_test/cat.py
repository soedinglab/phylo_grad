"""
    Example implementation of a continuous CAT Model used for the benchmark test
"""

import torch

def rate_matrix(shared, energies):
    """
        shared is a [190] tensor representing the upper diagonal of the log(S) matrix
        energies is a [L, 20] tensor representing the energy of each amino acid at each side. The distribution is given by the softmax of the energies
    """
    dtype = shared.dtype
    S = torch.zeros((20,20), dtype=dtype, device=shared.device)
    S[*torch.triu_indices(20,20, offset= 1)] = shared
    S = torch.exp(S)
    
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(energies, dim=1))
    
    return S.unsqueeze(0).expand(energies.shape[0], -1, -1), sqrt_pi