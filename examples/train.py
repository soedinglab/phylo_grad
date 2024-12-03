import io
import torch
from torch.nn.functional import log_softmax
import numpy as np
import phylo_grad

# This file gives an example of how to use the phylo_grad library in a real world example
# It trains a variation of the CAT model

# This is everything defining the model
# It is a torch function which takes some parameters and outputs S and sqrt_pi
def rate_matrix(shared, energies):
    """
        shared is a [190] tensor representing the upper diagonal of the log(S) matrix
        energies is a [L, 20] tensor representing the energy of each amino acid at each side.
        The distribution is given by the softmax of the energies
    """
    dtype = shared.dtype
    S = torch.zeros((20,20), dtype=dtype)
    S[*torch.triu_indices(20,20, offset= 1)] = shared
    S = torch.exp(S)
    sqrt_pi = torch.sqrt(torch.nn.functional.softmax(energies, dim=1))
    return S.unsqueeze(0).expand(energies.shape[0], -1, -1), sqrt_pi

L = 2 # Number of sides

#Init random parameters
shared = torch.rand(190, requires_grad=True)
energies = torch.rand(L, 20, requires_grad=True)

# The tree topology:
tree_top = "(A:0.1, B:0.2, (C:0.5,D:0.2):0.1);"

# Create random leaf probabilities. Typically this would be a one hot of the sequence.
# This allows for maximum flexibility, for example gap treatment
leaf_log_p = { seq : log_softmax(torch.randn([L, 20]), dim=1) for seq in ['A', 'B', 'C', 'D']}

# Create the tree object
tree = phylo_grad.FelsensteinTree.from_newick(io.StringIO(tree_top), leaf_log_p, dtype=np.float32)
        
optimizer = torch.optim.Adam([shared, energies], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi = rate_matrix(shared, energies)
    
    # Calculate the gradients with respect to S and sqrt_pi
    result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
    
    # Backpropagate the gradients to our original parameters and invert the sign to maximize the likelihood
    S.backward(-torch.tensor(result['grad_s']))
    sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi']))
    
    # Print the current likelihood of the tree
    print(result['log_likelihood'].sum())
    
    optimizer.step()    