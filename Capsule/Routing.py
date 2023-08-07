import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

EPS = 10e-6

def safe_norm(s, dim=-1, keepdim=False):
    '''
    Calculate norm of capsule
    s: unormalized capsule
    '''
    squared_norm = (s ** 2).sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(squared_norm + EPS)

def squash(s, dim=-1):
    '''
    Calculate non-linear squash function
    s: unormalized capsule
    v = (|s|^2)/(1+|s|^2)*(s/|s|)
    '''
    norm = safe_norm(s, dim=dim, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2)
    v = scale * s / norm
    return v

def power_squash(s, dim=-1, m=6):
    '''
    Calculate non-linear squash function
    s: unormalized capsule
    v = (|s|^m)*(s/|s|)
    '''
    norm = safe_norm(s, dim=dim)
    scale = norm ** m
    v = scale * s / norm
    return v

def max_min_norm(s, dim=-1):
    norm = safe_norm(s, dim=dim, keepdim=True)
    max_norm, _ = torch.max(norm, dim=1, keepdim=True)
    min_norm, _ = torch.min(norm, dim=1, keepdim=True)
    return s / (max_norm - min_norm + EPS)

class CapsuleRouting(nn.Module):
    '''
    Routing Between Capsules 
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Args:
        mode: routing mode
        iters: number of iterations
        lam, m: parameters for routing method
    Input:
        u: pose     (b, B, C, P*P, h, w)
        a_in: activation (b, B, h, w)
    Output:
        v:        (b, C, P*P, h, w)
        a_out:     (b, C, h, w)
        'b = batch_size'
        'h, w = spatial feature size
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    '''
    def __init__(self, B, C, P, projection_type, cap_style, routing):

        super(CapsuleRouting, self).__init__()

        self.iters = routing['iters']
        self.mode = routing['type']
        self.temp = routing['temp']  # for fuzzy routing
        self.cap_style = cap_style
        self.projection_type = projection_type
        self.P = P
        self.B = B
        self.C = C
        # self._lambda = lam # for fuzzy and EM routing

        fan_in = self.B * self.P * self.P # in_caps types * receptive field size
        std = np.sqrt(2.) / np.sqrt(fan_in)
        bound = np.sqrt(3.) * std
        if(self.projection_type == '3D'):
            self.W_ij_list = nn.ModuleList([nn.Conv3d(1, self.C * self.P * self.P, 
            kernel_size=(self.P * self.P, 3, 3), stride=(self.P * self.P, 1, 1), padding=(0, 1, 1))
                        for i in range(self.B)])
        elif(self.projection_type == 'linear'):
            # Out ← [1, B * K * K, C, P, P] noisy_identity initialization
            self.W_ij = nn.Parameter(torch.clamp(1.*torch.eye(self.P,self.P).repeat( \
                self.B, self.C, 1, 1).permute(0, 2, 3, 1) \
                + torch.empty(self.B, self.P, self.P, self.C).uniform_(-bound, bound)
                , max=1))
        else:
            raise NotImplementedError

        for i in range(self.B):
            nn.init.kaiming_uniform_(self.W_ij_list[i].weight, a=np.sqrt(5))
        
    #     self.projection = nn.Conv3d(1, 1, kernel_size=(self.P * self.P, 1, 1))
    
    # def linear_project(self, u):
    #     '''
    #     '''
    #     b, C, P, h, w = u.shape
    #     capsule_projection = torch.concat([self.projection(x_sub) for x_sub in torch.split(u, 1, dim=1)], dim=1)
    #     return capsule_projection.squeeze(2)

    def zero_routing(self, u, a):
        return u, a

    def max_min_routing(self, u):
        
        b, B, C, P, h, w = u.shape
        c = torch.ones((b, B, C, 1, h, w), device=u.device)
        r = torch.zeros((b, B, C, 1, h, w), device=u.device)

        for i in range(self.iters):
            ## c <- (b, B, C, 1, f, f)
            # v = torch.sum(c * u, dim=1, keepdim=True)
            # import pdb; pdb.set_trace()
            v = squash(torch.sum(c * u, dim=1, keepdim=True), dim=3) #non-linear activation of weighted sum v = sum(c*u)
            ## v <- (b, 1, C, P * P, f, f)
            
            if i != self.iters - 1:
                r = r + torch.sum(u * v, dim=3, keepdim=True) #consine similarity u*v
                ## r <- (b, B, C, 1, f, f)
                max_r, _ = torch.max(r, dim=2, keepdim=True)
                min_r, _ = torch.min(r, dim=2, keepdim=True)
                c = (r - min_r)/(max_r - min_r) #c_ij = p + (b - min(b))/(max(b) - min(b))*(q - p)

        v = v.squeeze(1)
        # a_out = self.linear_project(v)
        a_out = safe_norm(v, dim=2)
        return v, a_out
    
    def dynamic(self, u):
        '''
        Implement as shown in the paper "Dynamic Routing between Capsules"
        '''
        ## r <- (b, B, C, 1, f, f)
        b, B, C, P, h, w = u.shape
        r = torch.zeros((b, B, C, 1, h, w), device=u.device)
      
        for i in range(self.iters):
            c = torch.softmax(r, dim=2) #c_ij = exp(r_ij)/sum_k(exp(r_ik))
            ## c <- (b, B, C, 1, f, f)
            v = squash(torch.sum(c * u, dim=1, keepdim=True), dim=3) #non-linear activation of weighted sum v = sum(c*u)
            ## v <- (b, 1, C, P * P, f, f)
            if i != self.iters - 1:
                r = r + torch.sum(u * v, dim=3, keepdim=True) #consine similarity u*v
                ## r <- (b, B, C, 1, f, f)

        v = v.squeeze(1)
        # a_out = self.linear_project(v)
        a_out = safe_norm(v, dim=2)
        return v, a_out
        
    def EM(self, u, a):
        '''
        Implement as shown in the paper "Matrix Capsules with EM Routing"
        '''
    
        ln_2pi = 0.5*np.log(2*np.pi)
        ## r <- (b, B, C, 1, f, f)
        b, B, C, P, h, w = u.shape
        r = torch.ones((b, B, 1, 1, h, w), device=a.device)
        a = a.unsqueeze(2).unsqueeze(3)
       
        for iter_ in range(self.iters):
            #E step
            if iter_ > 0:
                ln_pj = -1. * torch.sum((u - mu)**2 / (2 * sigma_sq), dim = 3, keepdim=True) \
                    - torch.sum(0.5 * torch.log(sigma_sq), dim = 3, keepdim=True) \
                    - ln_2pi
                # r <- (b, B, C, 1, f, f)
                r = torch.softmax(ln_pj + torch.log(a), dim=2)
                
            #M step
            r = r * a
            r_sum = torch.sum(r, dim=1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + EPS)
            # mu <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            # sigma <- (b, 1, C, P, f, f)
            sigma_sq = torch.sum(coeff * (u - mu)**2, dim=1, keepdim=True) + EPS
            cost_h = torch.sum(0.5*torch.log(sigma_sq) * r_sum, dim=3, keepdim=True)
            # a_out <- (b, 1, C, 1, f, f)
            # print(cost_h)
            a = torch.sigmoid(- 0.01 * cost_h)
        
        return mu.squeeze(1), a.squeeze(1).squeeze(2)
    
    def fuzzy(self, u, a):
        '''
        Implement as shown in the paper "Capsule Network with Shortcut Routing"
        '''
        m = self.temp
        b, B, C, P, h, w = u.shape
        r = torch.ones((b, B, C, 1, h, w), device=a.device)
        a = a.unsqueeze(2).unsqueeze(3)
     
        for iter_ in range(self.iters):
            #fuzzy coeff
            if iter_ > 0:
                r_n = safe_norm(u - mu, dim=3, keepdim=True) ** (2. / (m - 1))
                r_d = torch.sum(1. / r_n, dim=2, keepdim=True)
                # r <- (b, B, C, 1, f, f)
                r = (1. / (r_n * r_d)) ** m 
                
            #update pose
            r_sum = torch.sum(a * r, dim=1, keepdim=True) # TODO, dim=1?
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + EPS)
            # v <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            #calcuate activation
            # a <- (b, 1, C, 1, f, f)
            a = torch.sigmoid(r_sum)
      
        return mu.squeeze(1), a.squeeze(1).squeeze(2)        
    
    def forward(self, p, a):

        if self.projection_type == 'linear':
            # B times of 3D convolutions of shape (C * P, 1, 1) to get B x C votes
            pre_votes = [transform(x_sub) for transform, x_sub in zip(self.W_ij_list, torch.split(p, 1, dim=1))]
            u = torch.concat([vote.unsqueeze(1) for vote in pre_votes], dim=1)
            b, B, C, P, h, w = u.shape
            u = u.reshape(-1, B, self.C, self.P * self.P, h, w)
        elif self.projection_type == '3D':
            b, B, P, h, w = p.shape
            p = p.reshape(-1, self.B, self.P, self.P, h, w)
            # Multiplying with Transformations weights matrix
            u = torch.einsum('bBijHW, BjkC -> bBCikHW', p, self.W_ij)
            u = u.reshape(-1, self.B, self.C, self.P * self.P, h, w)
       
        if self.mode == "dynamic":
            v, a = self.dynamic(u)
        elif self.mode == "em":
            v, a = self.EM(u, a)
        elif self.mode == "fuzzy":
            v, a = self.fuzzy(u, a)
        elif self.mode == "max-min":
            v, a = self.max_min_routing(u)
        else:
            v, a = self.zero_routing(u, a)
        return v, a