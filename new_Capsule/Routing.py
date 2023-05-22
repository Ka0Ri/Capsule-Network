"""
Implement Routing methods: EM, Fuzzy, Dynamic
Authors: dtvu1707@gmail.com
"""

import numpy as np
from torch import nn
import torch
import math

def squash(s, dim=-1):
    """
    Calculate non-linear squash function
    s: unormalized capsule
    v = (|s|^2)/(1+|s|^2)*(s/|s|)
    """
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    v = scale * s / torch.sqrt(squared_norm + 10e-6)
    # v = s / torch.sqrt(squared_norm)
    return v

def stable_squash(u, dim = -1):
    norm = torch.norm(u, dim=dim, keepdim=True)
    scale = torch.softmax(norm, dim=2)
    u = scale * u / (norm + 10e-6)

    return u

def max_min_norm(s, dim=-1):
    norm = torch.norm(s, dim=dim, keepdim=True)
    max_norm, _ = torch.max(norm, dim=1, keepdim=True)
    min_norm, _ = torch.min(norm, dim=1, keepdim=True)
    return s / (max_norm - min_norm)

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
    def __init__(self, B, C, P=4, mode="dynamic", iters=3, lam=0.001, m=1.5):
        super().__init__()

        assert mode in  ["dynamic", "em", "fuzzy"], "routing method is not supported"
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
            self.beta_a = nn.Parameter(torch.ones(C, 1, 1, 1))
            self.beta_u = nn.Parameter(torch.zeros(C, 1, 1, 1))

    def dynamic(self, u, a):
        
        # u = stable_squash(u, dim=3)
        u = max_min_norm(u, dim=3)
        ## r <- (b, B, C, 1, f, f)
        b, B, C, P, f, f = u.shape
        r = torch.zeros((b, B, C, 1, f, f), device=a.device)
        a_in = torch.softmax(a, dim=1).unsqueeze(2).unsqueeze(3) * self.C

        for i in range(self.iters):
            c = torch.softmax(r, dim=2) * a_in
            ## c <- (b, B, C, 1, f, f)
            v = squash(torch.sum(c * u, dim=1, keepdim=True), dim=3) #non-linear activation of weighted sum v = sum(c*u)
            # v = torch.sum(c * u, dim=1, keepdim=True)
            ## v <- (b, 1, C, P * P, f, f)
            if i != self.iters - 1:
                r = r + torch.sum(u * v, dim=3, keepdim=True) #consine similarity u*v
                ## r <- (b, B, C, 1, f, f)

        a_out = torch.norm(v, dim=3)
        return v.squeeze(), a_out.squeeze()
        
    def EM(self, u, a):

        u = max_min_norm(u, dim=3)

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
                r = torch.softmax(ln_ap, dim=2) * a_out
                
            #M step
            r = r / (r.sum(dim=2, keepdim=True) + self._eps)
            r_sum = r.sum(dim=1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + self._eps)
            # mu <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            # sigma <- (b, 1, C, P, f, f)
            sigma_sq = torch.sum(coeff * (u - mu)**2, dim=1, keepdim=True)
            # print(sigma_sq[0])
            cost_h = (self.beta_u + 0.5*torch.log(sigma_sq + self._eps)) * r_sum
            # a_out <- (b, 1, C, 1, f, f)
            a_out = torch.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=3, keepdim=True)))

        return mu.squeeze(), a_out.squeeze()
    
    def fuzzy(self, u, a):

        u = max_min_norm(u, dim=3)

        r = a.unsqueeze(2).expand(-1, -1, self.C, -1, -1) * (1./self.C)
        r = r.unsqueeze(3) 

        for iter_ in range(self.iters):
            #fuzzy coeff
            if(iter_ > 0):
                r_n = (torch.norm((u - v), dim=3, keepdim=True)) ** (2/(self.m - 1))
                r_d = torch.sum(1. / (r_n + self._eps), dim=2, keepdim=True)
                # r <- (b, B, C, 1, f, f)
                r = (1. / (r_n * r_d + self._eps)) ** (self.m)
                
            #update pose
            r_sum = r.sum(dim = 1, keepdim=True)
            # # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + self._eps)
            # v <- (b, 1, C, P, f, f)
            v = torch.sum(coeff * u, dim=1, keepdim=True)

        #calcuate activation
        # a <- (b, 1, C, 1, f, f)
        a = torch.sigmoid((math.e - torch.log(r_sum)))
        return v.squeeze(), a.squeeze()
    
    def forward(self, u, a):

        if(self.mode == "dynamic"):
            v, a = self.dynamic(u, a)
        elif(self.mode == "em"):
            v, a = self.EM(u, a)
        elif(self.mode == "fuzzy"):
            v, a = self.fuzzy(u, a)
        return v, a
    
