"""
Capsules Layers

PyTorch implementation by Vu.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


############EM Routing################
def caps_em_routing(v, a_in, beta_u, beta_a, iters):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, 1)
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
            ln_ap = ln_pjh.sum(dim=5) + torch.log(a_out.view(b, h, w, 1, C))
            r = F.softmax(ln_ap, dim=4)
           
        #M step
        r_sum = r.sum(dim=3, keepdim=True)
        coeff = r / (r_sum + eps)
        
        coeff = coeff.view(b, h, w, B, C, 1)
        mu = torch.sum(coeff * v, dim=3, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=3, keepdim=True) + eps
        r_sum = r_sum.view(b, h, w, 1, C, 1)
        cost_h = (beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        a_out = torch.sigmoid(_lambda*(beta_a - cost_h.sum(dim=5, keepdim=True)))
        
    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
  
    return mu, a_out

def caps_EMsim_routing(v, a_in, beta_u, iters):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, 1)
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
    b, h, w, B, C, psize = v.shape

    r = torch.cuda.FloatTensor(b, h, w, B, C, 1).fill_(1./C)
    a_in = a_in.view(b, h, w, B, 1, 1)
    for iter_ in range(iters):
        r_sum = r.sum(dim=3, keepdim=True)
        nr = r / (r_sum + eps)

        r_ = nr * a_in
        r_sum = torch.sum(r_, dim=3, keepdim=True)
        mu = torch.sum(r_ * v, dim=3, keepdim=True)/(r_sum + eps)
      
        r_d = torch.sum(torch.sum(nr, dim=3, keepdim=True), dim=4, keepdim=True)
        pi = torch.sum(nr, dim=3, keepdim=True)/(r_d + eps)
        d = torch.sum(torch.abs(v - mu), dim=5, keepdim=True)
        r = pi*torch.exp(-d/2)
        
    d =  torch.sum(torch.abs(v - beta_u*mu), dim=5, keepdim=True)
    a_out = torch.sum(r_*torch.exp(-d/2), dim=3, keepdim=True)
    a_out = F.softmax(a_out, dim=4)
    # a_out = torch.sigmoid(a_out)

    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
  
    return mu, a_out

def caps_MS_routing(v, a_in, beta_u, iters):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, 1)
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
    b, h, w, B, C, psize = v.shape

    r = torch.cuda.FloatTensor(b, h, w, B, C, 1).fill_(1./C)
    mu = torch.mean(v, dim=3, keepdim=True)
    a_in = a_in.view(b, h, w, B, 1, 1)
    for iter_ in range(iters):
        r_sum = r.sum(dim=3, keepdim=True)
        nr = r / (r_sum + eps)
        
        k = torch.exp(-torch.sum(torch.abs(v - mu), dim=5, keepdim=True)/2)
        r_ = -1/2* nr * a_in * k
        r_sum = torch.sum(r_, dim=3, keepdim=True)
        mu = torch.sum(r_ * v, dim=3, keepdim=True)/(r_sum + eps)
      
        r = r + a_in*k

    d =  torch.sum(torch.abs(v - beta_u*mu), dim=5, keepdim=True) 
    a_out = torch.sum(nr*a_in*torch.exp(-d/2), dim=3, keepdim=True)
    # a_out = F.softmax(a_out, dim=4)
    a_out = torch.sigmoid(a_out)

    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
  
    return mu, a_out
############EM Routing################




class PoolingCaps(nn.Module):
    """
    Args:
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
    Shape:
        input:  (*, h,  w, B, P*P) & (*, h,  w, B, 1)
        output: (*, h', w', B, P*P) & (*, h,  w, B, 1)
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, K=3, P=4, stride=2, mode="average"):
        super(PoolingCaps, self).__init__()
        self.K = K
        self.P = P
        self.mode = mode
        self.psize = P*P
        self.stride = stride

    def pactching(self, x, K, stride):
        """
        Preparing windows for computing convolution
        Shape:
            Input:     (b, H, W, B, -1)
            Output:    (b, H', W', K, K, B, -1)
        """
        b, h, w, B, psize = x.shape
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) for k_idx in range(0, K)] for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :, :]
        x = x[:, :, :, idxs, :, :]
        x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
        return x, oh, ow

    def forward(self, x, a):
        b, h, w, B, psize = x.shape
        # patching
        p_in, oh, ow = self.pactching(x, self.K, self.stride)
        a_in, _, _ = self.pactching(a, self.K, self.stride)
        p_in = p_in.view(b, oh, ow, self.K*self.K, B, psize)
        a_in = a_in.view(b, oh, ow, self.K*self.K, B, 1)
        if(self.mode == "average"):
            p_out = p_in.mean(dim = 3)
            a_out = a_in.mean(dim = 3)
        elif(self.mode == "max"):
            a_out, indices = a_in.max(dim=3)
            p_out = p_in.permute(0, 1, 2, 4, 3, 5).contiguous()
            p_out = p_out.view(b*oh*ow*B, self.K*self.K, psize)[np.arange(b*oh*ow*B), indices.view(-1),:]
            p_out = p_out.view(b, oh, ow, B, psize)

        return p_out, a_out


class PrimaryCaps(nn.Module):
    """
    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
    Shape:
        input:  (*, A, h, w)
        p: (*, h', w', B, P*P)
        a: (*, h', w', B, 1)
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.B = B

    def forward(self, x):
        p = self.pose(x)
        b, c, h, w, = p.shape
        p = p.permute(0, 2, 3, 1)
        p = p.view(b, h, w, self.B, -1)
        a = self.a(x)
        a = a.permute(0, 2, 3, 1)
        a = a.view(b, h, w, self.B, 1)
        a = torch.sigmoid(a)
        return p, a

class ConvCaps(nn.Module):
    """
    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.
    Shape:
        input:  (*, h,  w, B, P*P) & (*, h,  w, B, 1) 
        output: (*, h', w', C, P*P) & (*, h', w', C,  1)
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3, coor_add=False, w_shared=False, routing_mode="EM"):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        self.routing_mode = routing_mode
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(C,1))
        self.beta_a = nn.Parameter(torch.zeros(C,1))
        # the filter size is P*P*k*k
        self.weights = nn.Parameter(torch.randn(1, 1, 1, K*K*B, C, P, P))

    def pactching(self, x):
        """
        Preparing windows for computing convolution
        Shape:
            Input:     (b, H, W, B, -1)
            Output:    (b, H', W', K, K, B, -1)
        """
        K = self.K
        stride = self.stride
        b, h, w, B, psize = x.shape
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) for k_idx in range(0, K)] for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :, :]
        x = x[:, :, :, idxs, :, :]
        x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
        return x, oh, ow

    def voting(self, x):
        """
        Preparing capsule data point for EM routing, 
        For conv_caps:
            Input:     (b, h, w, K*K*B, P*P)
            Output:    (b, h, w, K*K*B, C, P*P)
        """
        C = self.C
        P = self.P
        b, h, w, B, psize = x.shape #b: fix dimmension, B, random dimmension
        x = x.view(b, h, w, B, 1, P, P)
        v = (x @ self.weights) #(b*H*W)*(K*K*B)*C*P*P
        v = v.view(b, h, w, B, C, P*P)
        return v

    def add_coord(self, v):
        """
        Shape:
            Input:     (b, H, W, B, C, P*P)
            Output:    (b, H, W, B, C, P*P)
        """
        b, h, w, B, C, psize = v.shape
        coor1 = torch.arange(h, dtype=torch.float32) / h
        coor2 = torch.arange(w, dtype=torch.float32) / w
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor1
        coor_w[0, 0, :, 0, 0, 1] = coor2
      
        v = v + coor_h + coor_w
        v = v.view(b, 1, 1, h*w*B, C, psize)
        return v

    def forward(self, x, a):
        b, h, w, B, psize = x.shape
        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a)
        p_in = p_in.view(b, oh, ow, self.K*self.K*self.B, self.psize)
        v = self.voting(p_in)

        if not self.w_shared:
            a_in = a_in.view(b, oh, ow, self.K*self.K*self.B, 1)
        else:
            a_in = a_in.view(b, 1, 1, oh*ow*self.B, 1)
            # coor_add
            if self.coor_add:
                v = self.add_coord(v)
        # em_routing
        if(self.routing_mode == "EM"):
            p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, self.iters)
        elif(self.routing_mode == "EMsim"):
            p_out, a_out = caps_EMsim_routing(v, a_in, self.beta_u, self.iters)
        elif(self.routing_mode == "MS"):
            p_out, a_out = caps_MS_routing(v, a_in, self.beta_u, self.iters)
        # print("EM ", torch.cuda.memory_allocated() / 1024**2)
        return p_out, a_out

class DepthWiseConvCaps(nn.Module):
    """
    Args:
        B: input number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
    Shape:
        input:  (*, h,  w, B, (P*P+1)) & (*, h,  w, B, 1)
        output: (*, h',  w', B, (P*P+1)) & (*, h',  w', B, 1)
        h', w' is computed the same way as convolution layer
    """
    def __init__(self, B=32, K=3, P=4, stride=2, iters=3, routing_mode = "EM"):
        super(DepthWiseConvCaps, self).__init__()
        self.B = B
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.routing_mode = routing_mode
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(B,1))
        self.beta_a = nn.Parameter(torch.zeros(B,1))
        # the filter size is P*P*k*k
        self.weights = nn.Parameter(torch.randn(1, 1, 1, K*K, B, P, P))

    def pactching(self, x):
        """
        Preparing windows for computing convolution
        Shape:
            Input:     (b, H, W, B, -1)
            Output:    (b, H', W', K, K, B, -1)
        """
        K = self.K
        stride = self.stride
        b, h, w, B, psize = x.shape
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) for k_idx in range(0, K)] for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :, :]
        x = x[:, :, :, idxs, :, :]
        x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
        return x, oh, ow

    def voting(self, x):
        """
        Preparing capsule data point for EM routing, 
        For conv_caps:
            Input:     (b, h, w, K*K, B, P*P)
            Output:    (b, h, w, K*K, B, P*P)
        """
        P = self.P
        b, h, w, K, B, psize = x.shape
        x = x.view(b, h, w, K, B, P, P)
        v = (x @ self.weights)
        v = v.view(b, h, w, K, B, P*P)
        return v

    def forward(self, x, a):
        b, h, w, B, psize = x.shape
        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a)
        p_in = p_in.view(b, oh, ow, self.K*self.K, self.B, self.psize)
        # voting
        v = self.voting(p_in)
        # print("voting ", torch.cuda.memory_allocated() / 1024**2)
        a_in = a_in.view(b, oh, ow, self.K*self.K, self.B)

        if(self.routing_mode == "EM"):
            p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, self.iters)
        elif(self.routing_mode == "EMsim"):
            p_out, a_out = caps_EMsim_routing(v, a_in, self.beta_u, self.iters)
        elif(self.routing_mode == "MS"):
            p_out, a_out = caps_MS_routing(v, a_in, self.beta_u, self.iters)
        # print("EM ", torch.cuda.memory_allocated() / 1024**2)
       
        return p_out, a_out


class FCCaps(nn.Module):
    """
    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        P: size of pose matrix is P*P
        iters: number of EM iterations
    Shape:
        input:  (*, B, P*P) &  (*, B, 1)
        output: (*, C, P*P) & (*, C, 1)
        h', w' is computed the same way as convolution layer
        parameter size is: B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, P=4, iters=3, routing_mode="EM"):
        super(FCCaps, self).__init__()
        self.B = B
        self.C = C
        self.P = P
        self.psize = P*P
        self.iters = iters
        self.routing_mode = routing_mode
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # the filter size is P*P*k*k
        self.weights = nn.Parameter(torch.randn(1, B, C, P, P))

    def voting(self, x):
        """
        Preparing capsule data point for EM routing, 
        Num Capsules at layer l: B
        Num Capsules at layer l + 1: C
        For conv_caps:
            Input:     (b, B, P*P)
            Output:    (b, B, C, P*P)
        """
        P = self.C
        C = self.P
        b, B, psize = x.shape
        #b: fix dimmension, B, random dimmension
        x = x.view(b, B, 1, P, P)
        v = x @ self.weights # b*B*C*P*P
        v = v.view(b, B, C, P*P)
        return v
    
    def forward(self, x, a):
        b, B, psize = x.shape
        
        # voting
        v = self.voting(x)
        v = v.view(b, 1, 1, self.B, self.C, self.psize)
        a_in = a.view(b, 1, 1, self.B, 1)

        if(self.routing_mode == "EM"):
            p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, self.iters)
        elif(self.routing_mode == "EMsim"):
            p_out, a_out = caps_EMsim_routing(v, a_in, self.beta_u, self.iters)
        elif(self.routing_mode == "MS"):
            p_out, a_out = caps_MS_routing(v, a_in, self.beta_u, self.iters)
        p_out = p_out.view(b, self.C, self.psize)
        a_out = a_out.view(b, self.C, 1)

        return p_out, a_out