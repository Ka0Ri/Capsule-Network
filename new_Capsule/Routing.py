"""
Implement Routing methods: EM, Fuzzy, Dynamic
Authors: dtvu1707@gmail.com
"""

import numpy as np
from torch import nn
import torch
from torch.autograd import Variable

def dot_product(a, b, dim=-1, keep_dims=True):

    return (a * b).sum(dim=dim, keepdim=keep_dims)

def squash(s, dim=-1):
    """
    Calculate non-linear squash function
    s: unormalized capsule
    v = (|s|^2)/(1+|s|^2)*(s/|s|)
    """
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    v = scale * s / torch.sqrt(squared_norm)
    return v


class CapsuleRouting(nn.Module):
    '''
    Routing Between Capsules 
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        u: pose     (b, B, C, P*P, f, f)
        a_in: activation (b, B, f, f)
    Output:
        v:        (b, C, P*P, f, f)
        a_out:     (b, C, f, f)
        'b = batch_size'
        'f' = spatial feature size
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    '''
    def __init__(self, B, C, P=4, mode="dynamic", iters=3, lam=0.01, m=2):
        super().__init__()

        assert mode in  ["dynamic", "attention", "em", "fuzzy"], "routing method is not supported"
        self.iters = iters
        self.C = C
        self.P = P
        self.B = B
        self._eps = 10e-6
        self._lambda = lam # for fuzzy and EM routing
        self.m = m # for fuzzy routing
        self.mode = mode

        # optional params
        if(self.mode == "em"):
            self.beta_a = nn.Parameter(torch.zeros(C, 1, 1, 1))
            self.beta_u = nn.Parameter(torch.zeros(C, 1, 1, 1))

    def dynamic(self, u, a):
        
        ## r <- (b, B, C, 1, f, f)
        r = a.unsqueeze(2).expand(-1, -1, self.C, -1, -1) * (1./self.C)
        r = r.unsqueeze(3)

        for i in range(self.iters):
            c = torch.softmax(r, dim=2)
            ## c <- (b, B, C, 1, f, f)
            v = squash(dot_product(c, u, dim=1), dim=3)#non-linear activation of weighted sum v = sum(c*u)
            ## v <- (b, 1, C, P * P, f, f)
            if i != self.iters - 1:
                r = r + dot_product(u, v, dim=3)#consine similarity u*v
                ## r <- (b, C, 1, 1, f, f)

        a_out = torch.norm(v, dim=3, keepdim=True)
        return v.squeeze(), a_out.squeeze()
    
    def attention(self, u, a):
        
        ## r <- (b, B, C, 1, f, f)
        r = a.unsqueeze(2).expand(-1, -1, self.C, -1, -1) * (1./self.C)
        r = r.unsqueeze(3)

        for i in range(self.iters):
            c = torch.softmax(r, dim=2)
            ## c <- (b, B, C, 1, f, f)
            v = dot_product(c, u, dim=1) # no activation
            ## v <- (b, 1, C, P * P, f, f)
            if i != self.iters - 1:
                r = r + dot_product(u, v, dim=3)#consine similarity u*v
                ## r <- (b, B, C, 1, f, f)
        a_out = torch.relu(r.sum(dim=1))
        return v.squeeze(), a_out.squeeze()
    
    def EM(self, u, a):

        ln_2pi = np.log(2*np.pi)
        ## r <- (b, B, C, 1, f, f)
        r = a.unsqueeze(2).expand(-1, -1, self.C, -1, -1) * (1./self.C)
        r = r.unsqueeze(3)
    
        for iter_ in range(self.iters):

            #E step
            if(iter_ > 0):
                ln_pjh = -1. * (u - mu)**2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt()) - 0.5*ln_2pi
                ln_ap = ln_pjh.sum(dim=3, keepdim=True) + torch.log(a_out)
                # r <- (b, B, C, 1, f, f)
                r = torch.softmax(ln_ap, dim=2)
                r = r * a_out
            
            #M step
            r_sum = r.sum(dim=1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + self._eps)
            # mu <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            # sigma <- (b, 1, C, P, f, f)
            sigma_sq = torch.sum(coeff * (u - mu)**2, dim=1, keepdim=True)
            cost_h = (self.beta_u + 0.5*torch.log(sigma_sq)) * r_sum
            # logit <- (b, 1, C, 1, f, f)
            logit = self._lambda*(self.beta_a - cost_h.sum(dim=3, keepdim=True)) 
            # a_out <- (b, 1, C, 1, f, f)
            a_out = torch.sigmoid(logit)

        return mu.squeeze(), a_out.squeeze()
    
    def fuzzy(self, u, a):

        r = a.unsqueeze(2).expand(-1, -1, self.C, -1, -1) * (1./self.C)
        r = r.unsqueeze(3)

        for iter_ in range(self.iters):
            #fuzzy coeff
            if(iter_ > 0):
                r_n = (torch.norm((u - v), dim=3, keepdim=True)) ** (2/(self.m - 1))
                r_d = torch.sum(1. / (r_n + self._eps), dim=2, keepdim=True)
                # r <- (b, B, C, 1, f, f)
                r = (1. / (r_n * r_d)) ** (self.m)
                
            #update pose
            r_sum = r.sum(dim = 1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + self._eps)
            # v <- (b, 1, C, P, f, f)
            v = torch.sum(coeff * u, dim=1, keepdim=True)

        #calcuate activation
        sigma_sq = torch.sum(coeff * (u - v) ** 2, dim=3, keepdim=True)
        # a <- (b, 1, C, 1, f, f)
        a = torch.mean( -0.5*torch.log(sigma_sq), dim=1, keepdim=True)
       
        return v.squeeze(), a.squeeze()
    
    def forward(self, u, a):

        if(self.mode == "dynamic"):
            v, a = self.dynamic(u, a)
        elif(self.mode == "attention"):
            v, a = self.attention(u, a)
        elif(self.mode == "em"):
            v, a = self.EM(u, a)
        elif(self.mode == "fuzzy"):
            v, a = self.fuzzy(u, a)
        return v, a
    

if __name__ == '__main__':

    u = torch.rand((2, 8, 2, 16, 3, 3))
    a_in = torch.rand((2, 8, 3, 3))
    beta = torch.rand((8, 1))
    lamda = torch.rand((8, 1))
    
    routing = CapsuleRouting(B= 8, C = 2, P = 4)
    v, a_out = routing(u, a_in)
    print(v.shape)
    print(a_out.shape)
    print(a_out)
