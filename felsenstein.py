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
            matrices will be [B,S,S]
        """
        # [a,b] := log p(x_m = b | x_n = a, t)
        rate = self.time * matrices
        mexp = torch.matrix_exp(rate)
        
        self.log_transitions = torch.log(mexp)
        
        if len(self.children) > 0: # Inner node
            self.precomp = torch.zeros([1,1], dtype=matrices.dtype) # this will be broadcasted to [B,S]
            
            for child in self.children:
                child.precompute(matrices)
                self.precomp = self.precomp + (child.precomp.unsqueeze(1) + child.transitions).logsumexp(dim = 2)

class FelsensteinTree:
    
    def __init__(self, tree):
        self.nodes = [FelsensteinNode(t) for _,t,_ in tree]
        
        for i, (parent, _, seq) in enumerate(tree):
            if seq:
                self.nodes[i].precomp = seq
            
            
            if parent == -1:
                self.root = self.nodes[i]
                continue
            self.nodes[parent].children.append(self.nodes[i])