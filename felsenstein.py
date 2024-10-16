import torch
import numpy as np

def g(x : torch.Tensor) -> torch.Tensor:
    # Because pytorch does not correctly handle unmasked Nan in backprob we have to make sure to never create one
    mask = torch.abs(x) < 1e-6
    x1 = torch.where(mask, x, 0.0) # Used for the small values, 0 will give 1 and not Nan
    x2 = torch.where(mask, 1.0, x) # Used for the big values, the small values will be set to 1 so they don't get Nan
    
    # Near 0 we use g(x) = 1 / (1 - x)
    return torch.where(mask, 1 / (1 - x1), 2 * x2 / (1 - torch.exp(-2 * x2)))

class FelsensteinNode:
    def __init__(self, time):
        self.children = []
        if time:
            self.time = max(time, 1e-4)
        else:
            self.time = 1e-4

    def compute(self, matrices):
        """
            matrices will be [L,S,S]
        """
        # [a,b] := log p(x_m = b | x_n = a, t)
        rate = self.time * matrices
        mexp = torch.matrix_exp(rate)
        
        self.log_transitions = torch.log(mexp)
        
        if len(self.children) > 0: # Inner node
            self.precomp = torch.zeros([1,1], dtype=matrices.dtype) # this will be broadcasted to [L,S]
            
            for child in self.children:
                child.compute(matrices)
                self.precomp = self.precomp + (child.precomp.unsqueeze(1) + child.log_transitions).logsumexp(dim = 2)

class FelsensteinTree:
    
    def __init__(self, tree):
        self.nodes = [FelsensteinNode(t) for _,t,_ in tree]
        
        for i, (parent, _, seq) in enumerate(tree):
            if seq is not None:
                self.nodes[i].precomp = seq
            
            
            if parent == -1:
                self.root = self.nodes[i]
                continue
            self.nodes[parent].children.append(self.nodes[i])
            
    def log_likelihood(self, S, sqrt_pi):
        matrices = rate_matrix_from_S(S, sqrt_pi)
        self.root.compute(matrices)
        
        root_likelihood = self.root.precomp
        
        return (root_likelihood + torch.log(sqrt_pi) * 2).logsumexp(dim = 1)
    

def rate_matrix_from_S(S, sqrt_pi):
    """
        Only upper halve of S is considered
    """
    S = torch.triu(S)
    S = S + S.transpose(1,2)
    rate = torch.diag_embed(1/sqrt_pi) @ S @ torch.diag_embed(sqrt_pi)
    return rate - torch.diag_embed(torch.sum(rate, axis=-1))