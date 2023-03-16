import numpy as np
import torch
from torch.autograd import Variable

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

##---EM Routing---

def caps_EM_routing(v, a_in, beta_u, beta_a, iters):
    """
    EM routing proposed in Procedure 1 paper: Matrix capsule with EM routing
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P) - pose
        a_in:      (b, h, w, B, 1) - activation
    Output:
        mu:        (b, h, w, C, P*P)
        a_out:     (b, h, w, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    eps = 1e-06
    _lambda = 1e-03
    ln_2pi = torch.cuda.FloatTensor(1).fill_(np.log(2*np.pi))
    b, h, w, B, C, psize = v.shape

    for iter_ in range(iters):
        #E step
        if(iter_ == 0):
            r = torch.cuda.FloatTensor(b, h, w, B, C).fill_(1./C) * a_in
        else:
            ln_pjh = -1. * (v - mu)**2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt()) - 0.5*ln_2pi
            a_out = a_out.view(b, h, w, 1, C)
            ln_ap = ln_pjh.sum(dim=5) + torch.log(a_out)
            r = torch.softmax(ln_ap, dim=4)
            r = r * a_out
           
        #M step
        r_sum = r.sum(dim=3, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, h, w, B, C, 1)
        mu = torch.sum(coeff * v, dim=3, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=3, keepdim=True) + eps
        r_sum = r_sum.view(b, h, w, 1, C, 1)
        cost_h = (beta_u + 0.5*torch.log(sigma_sq)) * r_sum
        logit = _lambda*(beta_a - cost_h.sum(dim=5, keepdim=True))
        a_out = torch.sigmoid(logit)
        
        
    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
  
    return mu, a_out

##---Dynamic Routing---

def caps_Dynamic_routing(u, b=None, iters=3):
    """
    Dynamic Routing in Procedure 1 in paper: Dynamic Routing Between Capsules 
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        u:         (b, h, w, B, C, P*P)
    Output:
        mu:        (b, h, w, C, P*P)
        a_out:     (b, h, w, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    batch, h, w, B, C, psize = u.shape
    if(b is None):
        b = Variable(torch.zeros(*u.size())).cuda()
    for i in range(iters):
        c = torch.softmax(b, dim=4)
        v = squash((c * u).sum(dim=3, keepdim=True))#non-linear activation of weighted sum v = sum(c*u)
        if i != iters - 1:
            b = b + (u * v).sum(dim=-1, keepdim=True)#consine similarity u*v
            
    a_out = torch.norm(v, dim=-1)
    a_out = torch.sigmoid(a_out)
    v = v.view(batch, h, w, C, psize)
    a_out = a_out.view(batch, h, w, C, 1)
    return v, a_out

##---Fuzzy Routing---

def caps_Fuzzy_routing(V, a_in, beta_a, _lambda, m, iters):
    """
    Fuzzy Routing in Fuzzy approach in paper: Capsule Network with Shortcut Routing
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P) - pose
        a_in:      (b, h, w, B, 1) - activation
    Output:
        mu:        (b, h, w, C, P*P)
        a_out:     (b, h, w, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    eps = 1e-06
    b, h, w, B, C, psize = V.shape
    for iter_ in range(iters):
        #fuzzy coeff
        if(iter_ == 0):
            r = torch.cuda.FloatTensor(b, h, w, B, C).fill_(1./C) * a_in
        else:
            r_n = (torch.norm((V - g), dim=-1)) ** (2/(m - 1)) + eps
            r_d = torch.sum(1. / (r_n), dim=4, keepdim=True)
            r = (1. / (r_n * r_d)) ** (m)
            
        #update pose
        r_sum = r.sum(dim = 3, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, h, w, B, C, 1)
        g = torch.sum(coeff * V, dim=3, keepdim=True)

    #calcuate probability
    sigma_sq = torch.sum(torch.sum(coeff * (V - g) ** 2, dim=-1, keepdim=True), dim=3, keepdim=True)
    a = torch.sigmoid(_lambda*(beta_a - 0.5*torch.log(sigma_sq)))

    g = g.view(b, h, w, C, psize)
    a = a.view(b, h, w, C, 1)
    return g, a

