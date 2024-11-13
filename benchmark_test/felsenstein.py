"""
    This is the Felsenstein implementation in PyTorch.
"""
import torch

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
        
        # To prevent negative values out of the matrix_exp which can happen due to numerical instability, we use the 1e-20 threshold
        self.log_transitions = torch.log(torch.max(mexp, torch.tensor(1e-20, dtype=mexp.dtype)))
        
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

def gradients(tree, S, sqrt_pi):
    S.requires_grad = True
    sqrt_pi.requires_grad = True
    
    logP = tree.log_likelihood(S, sqrt_pi)
    
    logP.sum().backward()
    
    return S.grad, sqrt_pi.grad