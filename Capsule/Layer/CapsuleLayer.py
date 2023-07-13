"""
Implement Routing methods: Dynamic, EM, Fuzzy
Implement Capsule Layers: (PrimaryCaps, ConvCap, EfficientCaps), Capsule Head
Authors: dtvu1707@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

EPS = 10e-6

def safe_norm(s, dim=-1):
    '''
    Calculate norm of capsule
    s: unormalized capsule
    '''
    squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
    return torch.sqrt(squared_norm + EPS)

def squash(s, dim=-1):
    '''
    Calculate non-linear squash function
    s: unormalized capsule
    v = (|s|^2)/(1+|s|^2)*(s/|s|)
    '''
    norm = safe_norm(s, dim=dim)
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
    norm = safe_norm(s, dim=dim)
    max_norm, _ = torch.max(norm, dim=1, keepdim=True)
    min_norm, _ = torch.min(norm, dim=1, keepdim=True)
    return s / (max_norm - min_norm)

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
    def __init__(self, B, C, P, routing):

        super(CapsuleRouting, self).__init__()

        self.iters = routing['iters']
        self.mode = routing['type']
        self.reduce = routing['reduce']
        self.temp = routing['temp']  # for fuzzy routing
        self.P = P
        self.B = B
        self.C = C
        # self._lambda = lam # for fuzzy and EM routing
    
        fan_in = self.B * self.P * self.P # in_caps types * receptive field size
        std = np.sqrt(2.) / np.sqrt(fan_in)
        bound = np.sqrt(3.) * std
        # Out ← [1, B * K * K, C, P, P] noisy_identity initialization
        self.W_ij = nn.Parameter(torch.clamp(1.*torch.eye(self.P,self.P).repeat( \
            self.B, self.C, 1, 1).permute(0, 2, 3, 1) \
            + torch.empty(self.B, self.P, self.P, self.C).uniform_(-bound, bound)
            , max=1))

    def zero_routing(self, u):

        # TODO: max_min_norm is not necessary, but we dont know
        # u = max_min_norm(u, dim=3)

        v = squash(torch.mean(u, dim=1, keepdim=True), dim=3)
        # v = torch.sum(u, dim=1, keepdim=True)
        a_out = safe_norm(v, dim=3)
        # a_out = torch.sigmoid(a_out)
        return v.squeeze(1), a_out.squeeze(1).squeeze(2)

    def max_min_routing(self, u):
        
        b, B, C, P, h, w = u.shape
        c = torch.ones((b, B, C, 1, h, w), device=u.device)
        r = torch.zeros((b, B, C, 1, h, w), device=u.device)
        for i in range(self.iters):
    
            ## c <- (b, B, C, 1, f, f)
            v = squash(torch.sum(c * u, dim=1, keepdim=True), dim=3) #non-linear activation of weighted sum v = sum(c*u)
            ## v <- (b, 1, C, P * P, f, f)
            if i != self.iters - 1:
                r = r + torch.sum(u * v, dim=3, keepdim=True) #consine similarity u*v
                ## r <- (b, B, C, 1, f, f)
                max_r, _ = torch.max(r, dim=2, keepdim=True)
                min_r, _ = torch.min(r, dim=2, keepdim=True)
                c = (r - min_r)/(max_r - min_r) #c_ij = p + (b - min(b))/(max(b) - min(b))*(q - p)

        a_out = safe_norm(v, dim=3)
        return v.squeeze(1), a_out.squeeze(1).squeeze(2)
    
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

        a_out = safe_norm(v, dim=3)
        return v.squeeze(1), a_out.squeeze(1).squeeze(2)
        
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
            cost_h = .01 * torch.sum(0.5*torch.log(sigma_sq) * r_sum, dim=3, keepdim=True)
            # a_out <- (b, 1, C, 1, f, f)
            # print(cost_h)
            a = torch.sigmoid(-cost_h)
        
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
                r_n = safe_norm(u - mu, dim=3) ** (2. / (m - 1))
                r_d = torch.sum(1. / r_n, dim=2, keepdim=True)
                # r <- (b, B, C, 1, f, f)
                r = (1. / (r_n * r_d)) ** m 
                
            #update pose
            r_sum = torch.sum(a * r, dim=1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + EPS)
            # v <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            #calcuate activation
            # a <- (b, 1, C, 1, f, f)
            a = torch.sigmoid(r_sum)
      
        return mu.squeeze(1), a.squeeze(1).squeeze(2)                               
    
    def forward(self, p, a):

        b, B, P, h, w = p.shape
        p = p.reshape(-1, self.B, self.P, self.P, h, w)
        # Multiplying with Transformations weights matrix
        u = torch.einsum('bBijHW, BjkC -> bBCikHW', p, self.W_ij)
        u = u.reshape(-1, self.B, self.C, self.P * self.P, h, w)

        if self.reduce:
            u = u.permute(0, 1, 4, 5, 2, 3)
            u = u.reshape(-1, self.B * h * w, self.C, self.P * self.P, 1, 1)
            a = a.reshape(-1, self.B * h * w, 1, 1)

        if self.mode == "dynamic":
            v, a = self.dynamic(u)
        elif self.mode == "em":
            v, a = self.EM(u, a)
        elif self.mode == "fuzzy":
            v, a = self.fuzzy(u, a)
        elif self.mode == "max_min":
            v, a = self.max_min_routing(u)
        else:
            v, a = self.zero_routing(u)
        return v, a

#--------------------Capsule Layer------------------------------------------------

class PrimaryCaps(nn.Module):
    """
    Primary Capsule Layer, implemented as paper Dynamic Routing between Capsules
    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        padding: padding of convolution
    Shape:
        input:  (*, A, h, w)
        p: (*, B, h', w', P*P)
        a: (*, B, h', w')
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A, B, K, stride, padding, P=4):
        super(PrimaryCaps, self).__init__()

        self.pose = nn.Sequential(nn.Conv2d(in_channels=A, out_channels=B*(P*P +1),
                            kernel_size=K, stride=stride, padding=padding),
                            )

        self.B = B
        self.P = P

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, sq=False):

        p = self.pose(x)
        b, d, h, w =  p.shape
        p = p.reshape(b, self.B, self.P ** 2 + 1, h, w)
        p, a = torch.split(p, [self.P **2, 1], dim=2)
        if sq:
            p = squash(p, dim=2)

        a = torch.sigmoid(a.squeeze(2))
        return p, a
    

class Caps_Dropout(nn.Module):
    '''
    Custom Dropout for Capsule Network
    The dropout will be perform at Capsule level
    '''
    def __init__(self, p: float = 0.5):
        super(Caps_Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            N, C, P, H, W = X.shape
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = binomial.sample([N, C, 1, H, W]).to(X.device)
            return X * mask * (1.0/(1-self.p))
        return X

    

class ShuffleBlock(nn.Module):
    '''
    Implementation in Pytorch
    '''
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.reshape(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W) 

class ProjectionHead(nn.Module):
    '''
    Projection Head for Primary Capsule
    args:
        in_channels: input channels
        out_channels: output channels
        n_layers: number of layers
        pooling: type of pooling
        TODO: add more oftion or just simply use?
    '''

    def __init__(self, in_channels, out_channels, shuffle=False, n_layers=2, pooling=None):
        super(ProjectionHead, self).__init__()

        in_channels = in_channels
        out_channels = out_channels
        n_layers = n_layers
        pooling = pooling
        k_size = 3
        p_size = 1

        self.projection = nn.Sequential()
        
        if(pooling == 'avg'):
            self.projection.add_module('avgpool', nn.AdaptiveAvgPool2d((1,1)))
            k_size = 1
            p_size = 0

        if(shuffle):
            self.projection.add_module('shuffle', ShuffleBlock(out_channels))
        
        if(n_layers == 1):
            self.projection.add_module('identity', nn.Identity())
        else:
            for i in range(n_layers):
                if(i == 0):
                    self.projection.add_module('conv'+str(i), nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=p_size))
                    self.projection.add_module('relu'+str(i), nn.ReLU())
                elif(i == n_layers - 1):
                    self.projection.add_module('conv'+str(i), nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=p_size))
                else:
                    self.projection.add_module('conv'+str(i), nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=p_size))
                    self.projection.add_module('relu'+str(i), nn.ReLU())

    def forward(self, x):
        return self.projection(x)


class AdaptiveCapsuleHead(nn.Module):
    '''
    Capsule Header combines of Primary Capsule and Linear Capsule.
    
    - Arguments:
        B: number of capsule in L layer
        head: head's configuration (check config file)
    '''
    def __init__(self, B, head):
        super(AdaptiveCapsuleHead, self).__init__()
    
        # self.reduce = head['caps']['reduce']
        n_layers = head['n_layers']
        n_emb = head['n_emb']
        pooling = head['caps']['pooling']
        shuffle = head['caps']['shuffle']
        self.P = head['caps']['cap_dims']
        self.cap_style = head['caps']['cap_style']
       
        assert n_emb % (self.P * self.P) == 0, "embedding is not divisible by P * P"
        assert B % (self.P * self.P) == 0, "channel is not divisible by P * P"

        self.primary_capsule = nn.Sequential()
        self.primary_capsule.add_module('projection', 
                                        ProjectionHead(B, n_emb, shuffle, n_layers, pooling))
        
        if(self.cap_style == 'hw'):
            self.B = B  * (n_layers == 1) + n_emb * (n_layers > 1)
            self.primary_capsule.add_module('pooling', nn.AdaptiveAvgPool2d((self.P, self.P)))
        elif(self.cap_style == 'c'):
            self.B = (B // (self.P ** 2)) * (n_layers == 1) + (n_emb // (self.P ** 2)) * (n_layers > 1) 
            if pooling == 'avg':
                self.primary_capsule.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))

        self.activation = nn.Sequential()
        self.activation.add_module('projection', ProjectionHead(B, self.B, shuffle, 2, pooling))
        if pooling == 'avg':
            self.activation.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
        self.activation.add_module('sigmoid', nn.Sigmoid())

        self.routinglayer = CapsuleRouting(self.B, head['n_cls'], head['caps']['cap_dims'], head['caps']['routing'])
          

    def forward(self, x, get_capsules=False):
        '''
        input: 
            tensor 4D (b, B, h, w)
        output:
            capsule 3D (b, C, P*P) / 5D (b, C, P*P, h, w)
            activation 2D (b, C) / 5D (b, C, h, w)
        '''
        # Primary capsule
        
        # p <- (b, B, P * P, h, w)
        # a <- (b, B, h, w)
        p = self.primary_capsule(x)
        a = self.activation(x)
        # x <- (b, C, h, w)
        b, d, h, w =  p.shape
        if(self.cap_style == 'hw'):
            p = p.reshape(b, d, self.P ** 2, 1, 1)
        elif(self.cap_style == 'c'):
            p = p.reshape(b, d // (self.P ** 2), self.P ** 2, h, w)
       
        p_out, a_out = self.routinglayer(p, a)
        # a_out = torch.log(a_out / (1 - a_out + EPS))
       
        if get_capsules:
            return p_out, a_out
        else: 
            return a_out