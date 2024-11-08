
import torch

import phylo_grad
import numpy as np

# This file gives an example of how to use the phylo_grad library in a real world example
# It trains a variation of the CAT model

# This is everything defining the model
# It is a torch function which takes some parameters and outputs S and sqrt_pi
def rate_matrix(shared, energies):
    """
        shared is a [190] tensor representing the upper diagonal of the log(S) matrix
        energies is a [L, 20] tensor representing the energy of each amino acid at each side. The distribution is given by the softmax of the energies
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

# The tree topology as nodes with parent and distance to parent
# The nodes are numbered from 0 to 3, where the root is 3
# The leaf nodes have to come before the internal nodes
# The root has to have parent -1
#
# Three nodes and one root
tree_top = np.array([[3,0.1], [3,0.2], [3,0.5], [-1,0.0]], dtype=np.float32)

# Create random leaf probabilities. Typically this would be a one hot of the sequence
leaf_log_p = torch.randn([L, 3, 20])
leaf_log_p = torch.nn.functional.log_softmax(leaf_log_p, dim = 2)

# The last parameter is the minimum edge length of the tree, for numerical reasons any edge shorter than this will be set to this value
tree = phylo_grad.FelsensteinTree(tree_top, leaf_log_p.numpy(), 1e-4)
        
optimizer = torch.optim.Adam([shared, energies], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    # This is the actual model part, where the parameters are mapped to S and sqrt_pi
    S, sqrt_pi = rate_matrix(shared, energies)
    
    # Calculate the gradients with respect to S and sqrt_pi
    result = tree.calculate_gradients(S.detach().numpy(), sqrt_pi.detach().numpy())
    
    # Backpropagate the gradients to our original parameters
    S.backward(-torch.tensor(result['grad_s']))
    sqrt_pi.backward(-torch.tensor(result['grad_sqrt_pi']))
    
    # Print the current likelihood of the tree
    print(result['log_likelihood'].sum())
    
    optimizer.step()
        