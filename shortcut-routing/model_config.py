
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import numpy as np
from utls import *

torch.cuda.set_device(training_settings['device'])
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device=torch.device(training_settings['device'])

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

############# Routing ##################################
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
            a_out = a_out.view(b, h, w, 1, C)
            ln_ap = ln_pjh.sum(dim=5) + torch.log(a_out)
            r = F.softmax(ln_ap, dim=4)
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

def caps_Dynamic_routing(u, b=None, iters=3):
    """
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
        c = F.softmax(b, dim=4)
        v = squash((c * u).sum(dim=3, keepdim=True))#non-linear activation of weighted sum v = sum(c*u)
        if i != iters - 1:
            b = b + (u * v).sum(dim=-1, keepdim=True)#consine similarity u*v
            
    a_out = torch.norm(v, dim=-1)
    a_out = torch.sigmoid(a_out)
    v = v.view(batch, h, w, C, psize)
    a_out = a_out.view(batch, h, w, C, 1)
    return v, a_out

def caps_Fuzzy_routing(V, a_in, beta_a, _lambda, m, iters):

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

def caps_Attention_routing(V, b=None, iters = 2):

    batch, h, w, B, C, psize = V.shape
    if(b is None):
        b = Variable(torch.zeros(*V.size())).cuda()
    for i in range(iters):        
        c = F.softmax(b, dim=4)     
        s = (V * c).sum(dim=3, keepdim=True)
        g = squash(s)
        b = (g * V).sum(dim=-1, keepdim=True)

    a = (g ** 2).sum(dim=-1) ** 0.5
    a = torch.sigmoid(a)

    g = g.view(batch, h, w, C, psize)
    a = a.view(batch, h, w, C, 1)
    return g, a

###########################################
########### New Capsule Net ###############

class CapLayer(nn.Module):
    """
       3D convolution to predict higher capsules
    """
    def __init__(self, num_in_caps, num_out_caps, kernel_size, stride=(1, 1), out_dim=(4,4), in_dim=(4,4), groups=1, bias=True):
        super(CapLayer, self).__init__()

        self.num_out_caps = num_out_caps
        self.out_dim = out_dim
        self.num_in_caps = num_in_caps
        self.in_dim = in_dim

        assert (in_dim[0] * in_dim [1]) % out_dim[0] == 0
        k = in_dim[0] * in_dim [1] // out_dim[0]

        self.W = nn.Conv3d(in_channels = num_in_caps, out_channels = num_out_caps * out_dim[1], 
            kernel_size = (k , kernel_size[0], kernel_size[1]), stride=(k, stride[0], stride[1]), groups=groups, bias=bias)

    def forward(self, x):
        """
            Input:
                x: (n, C, D, h, w) is input capsules
                C: number of capsules, D: dimmension of a capsule
                h, w: spatial size
            return:
                v: (n, C', D', h', w') is output capsules
        """
        s = self.W(x)
        n, C, D, h, w = s.size()
        s = s.view(n, self.num_out_caps, self.out_dim[0] * self.out_dim[1], h, w)
        
        return s


class ChannelwiseCaps(nn.Module):

    def __init__(self, num_in_caps, kernel_size, stride=(1,1), out_dim=(4,4), in_dim=(4,4)):
        super(ChannelwiseCaps, self).__init__()

        self.dw = CapLayer(num_in_caps, num_in_caps, kernel_size, stride, 
                            out_dim, in_dim, groups=num_in_caps, bias=False)

    def forward(self, x):
        prevoted_caps = self.dw(x)
 
        return prevoted_caps

class PointwiseCaps(nn.Module):

    def __init__(self, num_in_caps, num_out_caps):
        super(PointwiseCaps, self).__init__()

        self.pw = nn.Conv3d(num_in_caps, num_out_caps, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
    
        coarse_caps = self.pw(x)
        if(routing_settings['mode'] == 'attention' or
                routing_settings['mode'] == 'dynamic'):
            coarse_caps = squash(coarse_caps)
 
        return coarse_caps
    

class LocapBlock(nn.Module):
    """
       3D depth-wise convolution to predict higher capsules
    """
    def __init__(self, num_in_caps, num_out_caps, kernel_size, stride=(1,1), out_dim=(4,4), in_dim=(4,4)):
        super(LocapBlock, self).__init__()

        self.dw = CapLayer(num_in_caps, num_in_caps, kernel_size, stride, 
                            out_dim, in_dim, groups=num_in_caps, bias=False)
        # self.pw = CapLayer(num_in_caps, num_out_caps, kernel_size=(1,1), stride=(1,1), 
        #                     out_dim=out_dim, in_dim=in_dim, bias=False)
        self.pw = nn.Conv3d(num_in_caps, num_out_caps, kernel_size=1, stride=1, bias=False)


    def forward(self, x, sq=True):
        """
            Input:
                x: (n, C, D, h, w) is input capsules
                C: number of capsules, D: dimmension of a capsule
                h, w: spatial size
            return:
                v: (n, C', D', h', w') is output capsules
        """
        prevoted_caps = self.dw(x)
        coarse_caps = self.pw(prevoted_caps)
        if(sq == True):
            coarse_caps = squash(coarse_caps)
 
        return prevoted_caps, coarse_caps


class glocapBlockAttention(nn.Module):
    def __init__(self, in_channels, cap_dim, n_classes, settings, dim=(4,4)):
        super(glocapBlockAttention, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        n = self.n_classes * in_channels
        
        self.P = dim[0]
        
        self.weight = nn.Parameter(torch.randn(1, in_channels, 1, n_classes, self.P, self.P))
        self.n_routs = settings['n_rout']
        
    def forward(self, l, g):
        """
        Dynamic routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        -c: coefficients, attention (N, C_out, C_in)
        """
    
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        V = l[:, :, :, None, :, :] @ self.weight
        V = V.view(N, 1, 1, self.in_channels*W*H, self.n_classes, D)
        g = g[:, None, None, None, :, :]

        b = (g * V).sum(dim=-1, keepdim=True)   
        
        g, a = caps_Attention_routing(V, b, self.n_routs)

        g = g.squeeze()
        a = a.squeeze()
        return a, g

class glocapBlockDynamic(nn.Module):
    def __init__(self, in_channels, cap_dim, n_classes, settings, dim=(4,4)):
        super(glocapBlockDynamic, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        n = self.n_classes * in_channels
        
        self.P = dim[0]
        
        self.weight = nn.Parameter(torch.randn(1, in_channels, 1, n_classes, self.P, self.P))
        self.n_routs = settings['n_rout']
        
    def forward(self, l, g):
        """
        Dynamic routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        -c: coefficients, attention (N, C_out, C_in)
        """
    
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        V = l[:, :, :, None, :, :] @ self.weight
        V = V.view(N, 1, 1, self.in_channels*W*H, self.n_classes, D)
        g = g[:, None, None, None, :, :]

        b = (g * V).sum(dim=-1, keepdim=True)   
        
        g, a = caps_Dynamic_routing(V, b, self.n_routs)

        g = g.squeeze()
        a = a.squeeze()
        return a, g

class glocapBlockFuzzy(nn.Module):
    def __init__(self, in_channels, cap_dim, n_classes, settings, dim=(4,4)):
        super(glocapBlockFuzzy, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        n = self.n_classes * in_channels
        self.P = dim[0]
        self.eps = 1e-06
        self._lambda = settings['lambda']
        # self.pw = CapLayer(in_channels, n, kernel_size=(1,1),
        #         stride=(1,1), out_dim=dim, in_dim=dim, groups=in_channels)
        self.weight = nn.Parameter(torch.randn(1, in_channels, 1, n_classes, self.P, self.P))
        self.beta_a = nn.Parameter(torch.zeros(n_classes, 1))
        self.n_routs = settings['n_rout']
        self.m = settings['m']
        
    def forward(self, l, g):
        """
        EM routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        -c: coefficients, attention (N, C_out, C_in)
        """
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        V = l[:, :, :, None, :, :] @ self.weight
        V = V.view(N, 1, 1, self.in_channels*W*H, self.n_classes, D)
        g = g[:, None, None, None, :, :]

        #initialize activation probability
        r_n = (torch.norm((V - g), dim=-1)) ** (2/(self.m - 1)) + self.eps
        r_d = torch.sum(1. / (r_n), dim=4, keepdim=True)
        r = (1. / (r_n * r_d)) ** (self.m)

        g, a = caps_Fuzzy_routing(V, r, self.beta_a, self._lambda, self.m, self.n_routs)
        
        g = g.squeeze()
        a = a.squeeze()
        return a, g


class glocapBlockEM(nn.Module):
    def __init__(self, in_channels, cap_dim, n_classes, settings, dim=(4,4)):
        super(glocapBlockEM, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        n = self.n_classes * in_channels
        self.P = dim[0]
        self.eps = 1e-06
        self._lambda = settings['lambda']
        # self.pw = CapLayer(in_channels, n, kernel_size=(1,1),
        #         stride=(1,1), out_dim=dim, in_dim=dim, groups=in_channels)
        self.weight = nn.Parameter(torch.randn(1, in_channels, 1, n_classes, self.P, self.P))
        self.beta_a = nn.Parameter(torch.zeros(n_classes, 1))
        self.beta_u = nn.Parameter(torch.zeros(n_classes, 1))
        self.n_routs = settings['n_rout']
        
    def forward(self, l, g):
        """
        EM routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        -c: coefficients, attention (N, C_out, C_in)
        """
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        V = l[:, :, :, None, :, :] @ self.weight
        V = V.view(N, 1, 1, self.in_channels*W*H, self.n_classes, D)
        g = g[:, None, None, None, :, :]

        #initialize activation probability
        sigma_sq = torch.sum((V - g)**2, dim=3, keepdim=True) + self.eps
        cost_h = (self.beta_u + 0.5*torch.log(sigma_sq))
        logit = self._lambda*(self.beta_a - cost_h.sum(dim=-1, keepdim=True))
        a = torch.sigmoid(logit)
        a = a.view(N, 1, 1, 1, self.n_classes)

        g, a = caps_em_routing(V, a, self.beta_u, self.beta_a, self.n_routs)
        
        g = g.squeeze()
        a = a.squeeze()
        return a, g


class Caps_Dropout(nn.Module):
    """
    Custom Dropout for Capsule Network
    The dropout will be perform at Capsule level
    """
    def __init__(self, p: float = 0.5):
        super(Caps_Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            N, C, D, W, H = X.size()
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = binomial.sample([N, C, 1, W, H]).cuda()
            return X * mask * (1.0/(1-self.p))
        return X


class CoreArchitect(nn.Module):
    """

    """
    def __init__(self, input_channel=1, num_classes=10, n_routs=2):
        super(CoreArchitect, self).__init__()
        
        self.num_classes = num_classes
        self.primary_cap_num = architect_settings['PrimayCaps']['out']
        self.cap_dim = 4 * 4
        self.n_routs = n_routs
        
        self.conv_layers = []
        for i in range(architect_settings['n_conv']):
            conv = architect_settings['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        primary_caps = architect_settings['PrimayCaps']
        self.primary_cap = nn.Sequential(
            nn.Conv2d(primary_caps['in'], primary_caps['out'] * self.cap_dim, primary_caps['k'], primary_caps['s'], primary_caps['p']),
            nn.ReLU(),
            nn.BatchNorm2d(primary_caps['out'] * self.cap_dim),
            )

        self.dropout = Caps_Dropout(p=0.2)


        self.caps_layers = nn.ModuleList()
        self.dynamic_layers = nn.ModuleList()
        for i in range(architect_settings['n_caps']):
            caps = architect_settings['Caps' + str(i + 1)]
            self.caps_layers.append(ChannelwiseCaps(caps['in'], caps['k'], caps['s']))
            self.caps_layers.append(PointwiseCaps(caps['in'], caps['out']))

            if(routing_settings['mode'] == 'fuzzy'):
                self.dynamic_layers.append(glocapBlockFuzzy(caps['in'], self.cap_dim, num_classes, routing_settings['fuzzy']))
            elif(routing_settings['mode'] == 'attention'):
                self.dynamic_layers.append(glocapBlockAttention(caps['in'], self.cap_dim, num_classes, routing_settings['attention']))
            elif(routing_settings['mode'] == 'EM'):
                self.dynamic_layers.append(glocapBlockEM(caps['in'], self.cap_dim, num_classes, routing_settings['EM']))
            elif(routing_settings['mode'] == 'dynamic'):
                self.dynamic_layers.append(glocapBlockDynamic(caps['in'], self.cap_dim, num_classes, routing_settings['dynamic']))
    
        #kaiming initialization
        self.weights_init()
   
            
    def forward(self, x):
        
       
        x = self.conv_layers(x)
        x = self.primary_cap(x)
       
        n, c, h, w = x.size()
        l1 = x.view(n, self.primary_cap_num, self.cap_dim, h, w)
        l = self.dropout(l1)

        ls = [l]
        p_caps = []
        for i in range(0, 2 * architect_settings['n_caps'], 2):
            p_cap = self.caps_layers[i](l)
            p_caps.append(p_cap)
            l = self.caps_layers[i + 1](p_cap)
            ls.append(l)
           
        g = ls[-1].squeeze()
            
        for i in range(self.n_routs):
            for i in range(architect_settings['n_caps']):
                a, g = self.dynamic_layers[i](p_caps[i], g)

        return a
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

###########################################

###########################################
########### Old Capsule Net ###############



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
    def __init__(self, A=32, B=32, K=1, stride=1):
        P = 4
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=False)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=False)
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

        if(routing_settings['mode'] == 'dynamic'):
            p = squash(p)
       
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
    def __init__(self, B=32, C=32, K=3, stride=1):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = 4
        self.psize = self.P * self.P
        self.stride = stride
      
        # params
        # beta_u and beta_a are per capsule type,
        self.beta_u = nn.Parameter(torch.zeros(C,1))
        self.beta_a = nn.Parameter(torch.zeros(C,1))
        # the filter size is P*P*k*k
        self.weight = nn.Parameter(torch.randn(1, 1, 1, K*K*B, C, self.P, self.P))

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
        x = x.view(b, oh, ow, -1, psize)
        return x, oh, ow

    def voting(self, x):
        """
        Preparing capsule data points for routing, 
        For conv_caps:
            Input:     (b, h, w, K*K*B, P*P)
            Output:    (b, h, w, K*K*B, C, P*P)
        """
        C = self.C
        P = self.P
        b, h, w, B, psize = x.shape #b: fix dimmension, B, random dimmension
        x = x.view(b, h, w, B, 1, P, P)
        v = torch.matmul(x, self.weight) #(b*H*W)*(K*K*B)*C*P*P
        v = v.view(b, h, w, B, C, P*P)
        return v

    def forward(self, x, a):
        b, h, w, B, psize = x.shape
        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a) 
        v = self.voting(p_in)
       
        if(routing_settings['mode'] == 'EM'):
            p_out, a_out = caps_em_routing(v, a_in, self.beta_u, self.beta_a, routing_settings['EM']['n_rout'])
        elif(routing_settings['mode'] == 'dynamic'):
            p_out, a_out = caps_Dynamic_routing(v, iters=routing_settings['dynamic']['n_rout'])
        if(routing_settings['mode'] == 'fuzzy'):
            p_out, a_out = caps_Fuzzy_routing(v, a_in, self.beta_a, routing_settings['fuzzy']['lambda'], 
                            routing_settings['fuzzy']['m'] , routing_settings['fuzzy']['n_rout'])
        elif(routing_settings['mode'] == 'attention'):
            p_out, a_out = caps_Attention_routing(v, iters=routing_settings['attention']['n_rout'])
        
        return p_out, a_out

class CapNets(nn.Module):
    """
    """
    def __init__(self, input_channel=1, num_classes = 10):
        super(CapNets, self).__init__()

        self.conv_layers = []
        for i in range(architect_settings['n_conv']):
            conv = architect_settings['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        primary_caps = architect_settings['PrimayCaps']
        self.primary_caps = PrimaryCaps(primary_caps['in'], primary_caps['out'], primary_caps['k'])

        self.caps_layers = nn.ModuleList()
        for i in range(architect_settings['n_caps']):
            caps = architect_settings['Caps' + str(i + 1)]
            self.caps_layers.append(ConvCaps(caps['in'], caps['out'], caps['k'][0], caps['s'][0]))
           

    def forward(self, x):
       
        x = self.conv_layers(x)
        pose, a = self.primary_caps(x)

        for i in range(0, architect_settings['n_caps']):
            pose, a = self.caps_layers[i](pose, a)
           
        a = a.squeeze()
        pose = pose.squeeze()
   
        return a

###########################################




class MarginLoss(nn.Module):
    """
    Loss = T*max(0, m+ - |v|)^2 + lambda*(1-T)*max(0, |v| - max-)^2 + alpha*|x-y|
    """
    def __init__(self, num_classes, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
        # output, 128 x 10
        # print(output.size())
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False).cuda()
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        pos_part = F.relu(self.pos - output, inplace=True) ** 2
        neg_part = F.relu(output - self.neg, inplace=True) ** 2
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return loss.sum() / output.size(0)

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        return self.loss(output, target)
        
class SpreadLoss(nn.Module):

    def __init__(self, num_classes, m_min=0.1, m_max=0.9):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_classes = num_classes

    def forward(self, output, target, r=1):
        b, E = output.shape
        margin = self.m_min + (self.m_max - self.m_min)*r

        gt = torch.zeros(output.size(0), self.num_classes)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        gt = gt.bool()
        at = torch.masked_select(output, mask=gt)
        at = at.view(b, 1).repeat(1, self.num_classes)

        loss = F.relu(margin - (at - output), inplace=True)
        loss = loss**2
        loss = loss.sum() / b

        return loss

class ConvNeuralNet(nn.Module):
    def __init__(self, input_chanel = 1, num_classes = 10):
        super(ConvNeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_chanel, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(1024, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc(out)
        out = self.fc1(out)

        return out


