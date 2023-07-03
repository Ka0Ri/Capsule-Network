"""
Implement Routing methods: Dynamic, EM, Fuzzy
Implement Capsule Layers: (PrimaryCaps, ConvCap, EfficientCaps), Capsule Head
Authors: dtvu1707@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np

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
    scale = norm / (1 + norm)
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
    def __init__(self, mode='dynamic', iters=3, m=1.5):
        super().__init__()

        # assert mode in  ['dynamic', 'em', 'fuzzy', 'None'], "routing method is not supported"
        self.iters = iters
        # self._lambda = lam # for fuzzy and EM routing
        self.m = m # for fuzzy routing
        self.mode = mode

        self.vis = []

    def zero_routing(self, u, a):

        # u = max_min_norm(u, dim=3)
        v = torch.sum(u, dim=1, keepdim=True)
        a_out = safe_norm(v, dim=3)
        # a_out = torch.log_softmax(a_out, dim=2)
        return v.squeeze(1), a_out.squeeze(1).squeeze(2)

    
    def dynamic(self, u):
        '''
        Implement as shown in the paper "Dynamic Routing between Capsules"
        '''
        u = max_min_norm(u, dim=3)
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
        # u = max_min_norm(u, dim=3)

        ln_2pi = 0.5*np.log(2*np.pi)
        ## r <- (b, B, C, 1, f, f)
        b, B, C, P, h, w = u.shape
        r = torch.ones((b, B, 1, 1, h, w), device=a.device) * (1./C)
        a = torch.unsqueeze(torch.unsqueeze(a, dim=2), dim=3)

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
            a = torch.softmax(- cost_h, dim=2)

        return mu.squeeze(1), a.squeeze(1).squeeze(2)
    
    def fuzzy(self, u, a):
        '''
        Implement as shown in the paper "Capsule Network with Shortcut Routing"
        '''
        # u = max_min_norm(u, dim=3)

        b, B, C, P, h, w = u.shape
        r = torch.ones((b, B, C, 1, h, w), device=a.device) * (1./C)
        a = a.unsqueeze(2).unsqueeze(3)

        for iter_ in range(self.iters):
            #fuzzy coeff
            if iter_ > 0:
                r_n = safe_norm(u - mu, dim=3) ** (2. / (self.m - 1))
                r_d = torch.sum(1. / r_n, dim=2, keepdim=True)
                # r <- (b, B, C, 1, f, f)
                r = (1. / (r_n * r_d)) ** (self.m)
                
            #update pose
            r_sum = torch.sum(r * a, dim=1, keepdim=True)
            # coeff <- (b, B, C, 1, f, f)
            coeff = r / (r_sum + EPS)
            # v <- (b, 1, C, P, f, f)
            mu = torch.sum(coeff * u, dim=1, keepdim=True)
            #calcuate activation
            # a <- (b, 1, C, 1, f, f)
            a = torch.softmax(r_sum, dim=2)

        return mu.squeeze(1), a.squeeze(1).squeeze(2)
    
    def forward(self, u, a):

        if self.mode == "dynamic":
            v, a = self.dynamic(u)
        elif self.mode == "em":
            v, a = self.EM(u, a)
        elif self.mode == "fuzzy":
            v, a = self.fuzzy(u, a)
        else:
            v, a = self.zero_routing(u, a)
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
    

class ConvCaps(nn.Module):
    '''
    2D Capsule Convolutional Layer, implemented as 
    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        padding: padding of convolution
        weight_init: weight initialization for transformation parameters matrix
        *args: arguments for routing method (as Routing class)
    Shape:
        input:  (*, B, P*P, h, w) & (*, B, h, w) 
        output: (*, C, P*P, h', w') & (*, C, P*P,  h', w')
        f', f' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    '''
    def __init__(self, B, C, K=3, stride=1, padding=0, P=4, weight_init='xavier_uniform', *args):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.padding = padding
        self.psize = self.P * self.P
        self.stride = stride
          
        # the filter size is P*P*k*k
        self.W_ij = torch.empty(1, self.B * self.K * self.K, self.C, self.P, self.P)

        if weight_init.split('_')[0] == 'xavier':
            fan_in = self.B * self.K*self.K * self.psize # in_caps types * receptive field size
            fan_out = self.C * self.K*self.K * self.psize # out_caps types * receptive field size
            std = np.sqrt(2. / (fan_in + fan_out))
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init.split('_')[0] == 'kaiming':
            # fan_in preserves magnitude of the variance of the weights in the forward pass.
            fan_in = self.B * self.K*self.K * self.psize # in_caps types * receptive field size
            # fan_out has same affect as fan_in for backward pass.
            # fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2.) / np.sqrt(fan_in)
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))
        elif weight_init == 'noisy_identity' and self.psize > 2:
            b = 0.01 # U(0,b)
            # Out ← [1, B * K * K, C, P, P]
            self.W_ij = nn.Parameter(torch.clamp(.1*torch.eye(self.P,self.P).repeat( \
                1, self.B * self.K * self.K, self.C, 1, 1) + \
                torch.empty(1, self.B * self.K * self.K, self.C, self.P, self.P).uniform_(0,b), max=1))
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))
        
        assert len(args) == 3, "Wrong number of arguments"
        self.routinglayer = CapsuleRouting(mode=args[0], iters=args[1], m=args[2])


    def pactching(self, x):
        '''
        Preparing windows for computing convolution
        Shape:
            Input:     (b, B, -1, h, w)        
            Output:    (b, B, -1, h', w', K, K)
        '''
        if len(x.shape) < 5:
            x = x.unsqueeze(2)
    
        pd = (self.padding, ) * 4
        x = torch.nn.functional.pad(x, pd, "constant", 0)
        K = self.K
        stride = self.stride
        b, B, psize, f, f = x.shape
        oh = ow = int((f - K) / stride) + 1
        idxs = [[(h_idx + k_idx) for k_idx in range(0, K)] for h_idx in range(0, f - K + 1, stride)]
        x = x[:, :, :, idxs, :]
        x = x[:, :, :, :, :, idxs]
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        x = x.reshape(b, -1, psize, oh, ow)
        return x, oh, ow

    def voting(self, x):
        '''
        Preparing capsule data points for routing, 
        For conv_caps:
            Input:     (b, h, w, K*K*B, P*P)
            Output:    (b, h, w, K*K*B, C, P*P)
        '''
        C = self.C
        P = self.P
        b, B, psize, f, f = x.shape #b: fix dimmension, B, random dimmension
        x = x.reshape(b, B, P, P, f, f)
        v = torch.einsum('bBijhw, bBCjk -> bBCikhw', x, self.W_ij)
        v = v.reshape(b, B, C, psize, f, f)
        return v

    def forward(self, x, a):
        '''
        Forward pass
        '''
        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a)
        a_in = a_in.squeeze(2)
        v = self.voting(p_in)

        #Routing
        p_out, a_out = self.routinglayer(v, a_in)
        
        return p_out, a_out
    

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

    
class EffCapLayer(nn.Module):
    '''
       Efficient depth-wise convolution to predict higher capsules
    Input of a local block is a capsule at Lth layer
    Output of a local block is a capsule at (L+1)th layer
    Implemented by CapLayer module, so the arguments are set according to CapLayer's arguments
    This Layer contains 2 operations, Depth-wise convolution acts on each Capsule Styles (channels) without routing
    and Pointwise convolution acts on each unit with routing mechanism.
        B: number of capsule in L layer
        C: number of capsule in L + 1 layer
        K: 2D kernel size
        S: stride when doing convolution
        padding: padding when doing convolution
        P: size of a capsule
        groups: group convolution
        weight_init: weight init method ["xavier_uniform", "xavier_normal", 
                        "kaiming_normal", "kaiming_uniform", "noisy_identity"]
        args: arguments for routing method
        argv[0]: routing method
        argv[1]: number of iteration
        argv[2]: lambda for fuzzy/em routing
        argv[3]: m for fuzzy routing
    '''
    def __init__(self, B, C, K, S=1, padding=0, P=4, weight_init='xavier_uniform', *args):
        super(EffCapLayer, self).__init__()

        self.P = P
        self.C = C
        self.B = B
        self.psize = P * P

        self.dw = nn.Sequential(
            nn.Conv3d(in_channels = B, out_channels = B * P, kernel_size = (P , K, K), 
                      stride=(P, S, S), padding=(0, padding, padding), 
            groups=B, bias=False),
            nn.Sigmoid())

        self.a_dw = nn.Sequential(
            nn.Conv2d(in_channels=B, out_channels=B, kernel_size=K, 
                      stride=S, padding=padding, groups=B),
            nn.Sigmoid(),
        )
        
        self.W_ij = torch.empty(1, self.B, self.C, self.P, self.P)

        if weight_init.split('_')[0] == 'xavier':
            fan_in = self.B * self.psize # in_caps types * receptive field size
            fan_out = self.C * self.psize # out_caps types * receptive field size
            std = np.sqrt(2. / (fan_in + fan_out))
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init.split('_')[0] == 'kaiming':
            # fan_in preserves magnitude of the variance of the weights in the forward pass.
            fan_in = self.B * self.psize # in_caps types * receptive field size
            # fan_out has same affect as fan_in for backward pass.
            # fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            std = np.sqrt(2.) / np.sqrt(fan_in)
            bound = np.sqrt(3.) * std

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound))
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))
        elif weight_init == 'noisy_identity' and self.psize > 2:
            b = 0.01 # U(0,b)
            # Out ← [1, B * K * K, C, P, P]
            self.W_ij = nn.Parameter(torch.clamp(.1*torch.eye(self.P,self.P).repeat( \
                1, self.B, self.C, 1, 1) + \
                torch.empty(1, self.B, self.C, self.P, self.P).uniform_(0,b), max=1))
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))
        
        assert len(args) == 3, "Wrong number of arguments"
        self.routinglayer = CapsuleRouting(mode=args[0], iters=args[1], m=args[2])
    

    def forward(self, x, a):
        '''  
        Input:
            x (*, B, P, h, w): is input capsules
            a: activation (*, B, h, w)
            B: number of capsules, P * P: dimmension of a capsule
            h, w: spatial size
        return:
            p_out: (*, C, P*P, h', w') is output capsules
            a_out: (*, C, h', w')
        h', w' is computed the same way as convolution layer
        '''

        u = self.dw(x)
        b, B, P, h, w = u.shape
        B = B // self.P
        u = u.reshape(b, B, P, P, h, w)
        v = torch.einsum('bBijhw, bBCjk -> bBCikhw', u, self.W_ij)
        v = v.reshape(b, B, self.C, P * P, h, w)
       
        a_in = self.a_dw(a)
        #Routing
        p_out, a_out = self.routinglayer(v, a_in)
      
        return p_out, a_out

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

class AdaptiveCapsuleHead(nn.Module):
    '''
    Capsule Header combines of Primary Capsule and Linear Capsule.
    - mode:
        1: Backbone → Feature maps  → Adaptive Primary Capsule → Routing
        2: Backbone → AVGPool -> Feature maps → Split → Routing
        3: Backbone → AVGPool -> Feature maps → Split + Shuffle → Routing
        4: Backbone → AVGPool -> Feature maps → Primary Capsule → Routing
    - Arguments:
        B: number of capsule in L layer
        C: number of capsule in L + 1 layer
        get_capsules: Return Capsules's vector
        P: size of a capsule
        reduce: reduce the featrue maps before routing
        args: arguments for routing method
        argv[0]: routing method
        argv[1]: number of iteration
        argv[2]: m for fuzzy routing
    '''
    def __init__(self, B, C, P=4, mode=1, reduce=True,  *args):
        super(AdaptiveCapsuleHead, self).__init__()
        self.mode = mode
        self.P = P
        self.C = C
        self.reduce = reduce

        assert B % (P * P) == 0, "channel is not divisible by P * P"
        self.B = B // (P * P)

        if mode == 1:
            self.primary_capsule = nn.Sequential(nn.Conv2d(in_channels=B,
                            out_channels=self.B*(P*P +1), kernel_size=1, stride=1, padding=0)
                                    )
        elif mode == 2:
            self.primary_capsule = nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    ShuffleBlock(self.B)
                                    )
        elif mode == 3:
            self.primary_capsule = nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Identity()
                                    )
        elif mode == 4:
            self.primary_capsule = nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Conv2d(in_channels=B,
                            out_channels=self.B*(P*P +1), kernel_size=1, stride=1, padding=0),
                                    nn.ReLU(inplace=True)
                                    )


        self.W_ij = torch.empty(1, self.B, self.C, self.P, self.P)
        fan_in = self.B * self.P * self.P # in_caps types * receptive field size
        std = np.sqrt(2.) / np.sqrt(fan_in)
        bound = np.sqrt(3.) * std
        self.W_ij = nn.Parameter(self.W_ij.normal_(0, std))
        # Out ← [1, B * K * K, C, P, P] noisy_identity initialization
        # self.W_ij = nn.Parameter(torch.clamp(.1*torch.eye(self.P,self.P).repeat( \
        #     1, self.B, self.C, 1, 1) + \
        #     torch.empty(1, self.B, self.C, self.P, self.P).uniform_(-0.05,0.05), max=1))

        assert len(args) == 3, "Wrong number of arguments"
        self.routinglayer = CapsuleRouting(mode = args[0], iters=args[1], m=args[2])

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
        # x <- (b, C, h, w)
        b, d, h, w =  p.shape
        if self.mode == 1 or self.mode == 4:
            p = p.reshape(b, self.B, self.P ** 2 + 1, h, w)
            p, a = torch.split(p, [self.P **2, 1], dim=2)
            # p = squash(p, dim=2)
            a = a.squeeze(2)
            # a = torch.relu(a.squeeze(2))
        else:
            p = p.reshape(b, self.B, self.P ** 2, h, w)
            # p = squash(p, dim=2)
            a = safe_norm(p, dim=2)
            # a = torch.relu(a)

        # Routing
        u = p.reshape(b, self.B, self.P, self.P, h, w)
        # Multiplying with Transformations weights matrix
        v = torch.einsum('bBijhw, bBCjk -> bBCikhw', u, self.W_ij)
        v = v.reshape(b, self.B, self.C, self.P * self.P, h, w)

        # adaptive pooling
        if(self.reduce):
            v = v.permute(0, 1, 4, 5, 2, 3)
            v = v.reshape(b, -1, self.C, self.P * self.P, 1, 1)
            a = a.reshape(b, -1, 1, 1)
        
        p_out, a_out = self.routinglayer(v, a)
        # log softmax
        # a_out = torch.log(a_out)

        if get_capsules:
            return p_out, a_out
        else: 
            return a_out

        
        
    