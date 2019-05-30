"""
Capsules Layers

PyTorch implementation by Vu.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


############EM Routing################
def caps_em_routing_CPU(v, a_in, beta_u, beta_a, eps, iters, _lambda, ln_2pi):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, C, 1)
    Output:
        mu:        (b, h, w, B, C, P*P)
        a_out:     (b, h, w, B, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    b, h, w, B, C, psize = v.shape
    V_temp = v.data.cpu().numpy()
    a_temp = a_in.data.cpu().numpy()
    for iter_ in range(iters):
        #E step
        if(iter_ == 0):
            r = 1./C*np.ones((b, h, w, B, C)) * a_temp
            # print("r ", r.shape)
        else:
            ln_pjh = -(V_temp - mu)**2 / (2*sigma_sq) - np.log(np.sqrt(sigma_sq)) - 0.5*np.log(2*np.pi)
            # print("ln_pjh ", ln_pjh.shape)
            ln_ap = np.sum(ln_pjh, axis=5) + 1./B * np.log(r_sum)
            # print("ln_ap ", ln_ap.shape)
            exp_ln_ap = np.exp(ln_ap)
            # print("exp_ln_ap ", exp_ln_ap.shape)
            r = exp_ln_ap/(np.sum(exp_ln_ap, axis=4, keepdims=True))
            # print("r ", r.shape)
        if(iter_ < iters - 1):
            #M-step
            r_sum = np.sum(r, axis=3, keepdims=True)
            # print("r_sum ", r_sum.shape)
            coeff = r / (r_sum)
            # print("coeff ", coeff.shape)
            coeff = np.reshape(coeff, (b, h, w, B, C, 1))
            mu = np.sum((coeff * V_temp), axis=3, keepdims=True)
            # print("mu ", mu.shape)
            sigma_sq = np.sum((coeff * (V_temp - mu)**2), axis=3, keepdims=True)
            # print("sigma_sq ", sigma_sq.shape)

    #to cuda
    r_cuda = torch.from_numpy(r).float().cuda()
    r_sum_cuda = r_cuda.sum(dim=3, keepdim=True)
    coeff_cuda = r_cuda / (r_sum_cuda + eps)
    coeff_cuda = coeff_cuda.view(b, h, w, B, C, 1)
    # print("coeff", torch.cuda.memory_allocated() / 1024**2)
    mu_cuda= torch.sum(coeff_cuda * v, dim=3, keepdim=True)
    # print("mu", torch.cuda.memory_allocated() / 1024**2)
    sigma_sq_cuda = torch.sum(coeff_cuda * (v - mu_cuda)**2, dim=3, keepdim=True)
    # print("sigma_sq", torch.cuda.memory_allocated() / 1024**2)
    r_sum_cuda = r_sum_cuda.view(b, h, w, 1, C, 1)
    cost_h = (beta_u + torch.log(sigma_sq_cuda.sqrt())) * r_sum_cuda
    # print("cost_h ", torch.cuda.memory_allocated() / 1024**2)
    a_out = torch.sigmoid(_lambda*(beta_a - cost_h.sum(dim=5, keepdim=True)))
    # print("a_out ", torch.cuda.memory_allocated() / 1024**2)

    mu_cuda = mu_cuda.view(b, h, w, C, psize)
    a_out =  a_out.view(b, h, w, C, 1)
    return mu_cuda, a_out

def caps_em_routing(v, a_in, beta_u, beta_a, eps, iters, _lambda, ln_2pi):
    """
    i in B: layer l, j in h*w*C: layer l + 1
    with each cell in h*w, compute on matrix B*C
    Input:
        v:         (b, h, w, B, C, P*P)
        a_in:      (b, h, w, B, C, 1)
    Output:
        mu:        (b, h, w, B, C, P*P)
        a_out:     (b, h, w, B, C, 1)
        'b = batch_size'
        'h = height'
        'w = width'
        'C = depth'
        'B = K*K*B'
        'psize = P*P'
    """
    b, h, w, B, C, psize = v.shape

    for iter_ in range(iters):
        #E step
        if(iter_ == 0):
            r = torch.cuda.FloatTensor(b, h, w, B, C).fill_(1./C) * a_in
            # print("r init", torch.cuda.memory_allocated() / 1024**2)
        else:
            ln_pjh = -1. * (v - mu)**2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt()) - 0.5*ln_2pi
            # print("ln_pjh ", torch.cuda.memory_allocated() / 1024**2)
            ln_ap = ln_pjh.sum(dim=5) + torch.log(a_out.view(b, h, w, 1, C))
            # print("ln_ap ", torch.cuda.memory_allocated() / 1024**2)
            r = F.softmax(ln_ap, dim=4)
            # print("r ", torch.cuda.memory_allocated() / 1024**2)
        #M step
        r_sum = r.sum(dim=3, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, h, w, B, C, 1)
        # print("coeff", torch.cuda.memory_allocated() / 1024**2)
        mu = torch.sum(coeff * v, dim=3, keepdim=True)
        # print("mu", torch.cuda.memory_allocated() / 1024**2)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=3, keepdim=True)
        # print("sigma_sq", torch.cuda.memory_allocated() / 1024**2)
        r_sum = r_sum.view(b, h, w, 1, C, 1)
        cost_h = (beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # print("cost_h ", torch.cuda.memory_allocated() / 1024**2)
        a_out = torch.sigmoid(_lambda*(beta_a - cost_h.sum(dim=5, keepdim=True)))
        # print("a_out ", torch.cuda.memory_allocated() / 1024**2)
        
    mu = mu.view(b, h, w, C, psize)
    a_out = a_out.view(b, h, w, C, 1)
    return mu, a_out
############EM Routing################

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
        input:  (*, B, h,  w, (P*P+1))
        output: (*, C, h', w', (P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False):
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
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(np.log(2*np.pi))
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(C,1))
        self.beta_a = nn.Parameter(torch.zeros(C,1))
        # the filter size is P*P*k*k
        self.weights = nn.Parameter(torch.randn(1, 1, 1, K*K*B, C, P, P))

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

    def voting(self, x, Weights, C, P):
        """
        Preparing capsule data point for EM routing, 
        Num Capsules at layer l: KxKxB
        Num Capsules at layer l + 1: H*W*C
        For conv_caps:
            Input:     (b, h, w, K*K*B, P*P)
            Output:    (b, h, w, K*K*B, C, P*P)
        """
        b, h, w, B, psize = x.shape
        #b: fix dimmension, B, random dimmension
        x = x.view(b, h, w, B, 1, P, P)
        v = (x @ Weights) #(b*H*W)*(K*K*B)*C*P*P
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
        p_in, oh, ow = self.pactching(x, self.K, self.stride)
        a_in, _, _ = self.pactching(a, self.K, self.stride)
        # print("pactching ", torch.cuda.memory_allocated() / 1024**2)
        p_in = p_in.view(b, oh, ow, self.K*self.K*self.B, self.psize)
        v = self.voting(p_in, self.weights, self.C, self.P)
        # print("voting ", torch.cuda.memory_allocated() / 1024**2)
        if not self.w_shared:
            a_in = a_in.view(b, oh, ow, self.K*self.K*self.B, 1)
        else:
            a_in = a_in.view(b, 1, 1, oh*ow*self.B, 1)
            # coor_add
            if self.coor_add:
                v = self.add_coord(v)
        # em_routing
        p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, 
                                        self.eps, self.iters, self._lambda, self.ln_2pi)
        # print("EM ", torch.cuda.memory_allocated() / 1024**2)
        return p_out, a_out

class FCCaps(nn.Module):
    """
    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        P: size of pose matrix is P*P
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.
    Shape:
        input:  (*, B, (P*P+1))
        output: (*, C, (P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, P=4, iters=3):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.P = P
        self.psize = P*P
        self.iters = iters
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(np.log(2*np.pi))
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # the filter size is P*P*k*k
        self.weights = nn.Parameter(torch.randn(1, B, C, P, P))

    def voting(self, x, w, C, P):
        """
        Preparing capsule data point for EM routing, 
        Num Capsules at layer l: B
        Num Capsules at layer l + 1: C
        For conv_caps:
            Input:     (b, B, P*P)
            Output:    (b, B, C, P*P)
        """
        b, B, psize = x.shape
        #b: fix dimmension, B, random dimmension
        x = x.view(b, B, 1, P, P)

        v = x @ w # b*B*C*P*P
        v = v.view(b, B, C, P*P)
        return v
    
    def forward(self, x, a):
        b, B, psize = x.shape
        
        # voting
        v = self.voting(x, self.weights, self.C, self.P)
        v = v.view(b, 1, 1, self.B, self.C, self.psize)
        a_in = a.view(b, 1, 1, self.B, 1)

        p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, 
                                        self.eps, self.iters, self._lambda, self.ln_2pi)
        p_out = p_out.view(b, self.C, self.psize)
        a_out = a_out.view(b, self.C, 1)

        return p_out, a_out