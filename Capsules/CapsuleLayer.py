"""
Implement Capsule Layers: (PrimaryCaps, ConvCap), (LocapBlock, glocapBlock, CapLayer) - Shortcut Routing
Authors: dtvu1707@gmail.com
"""

import torch
import torch.nn as nn
from Routing import caps_Dynamic_routing, caps_EM_routing, caps_Fuzzy_routing, squash

#-----Conventional Layer------

class PrimaryCaps(nn.Module):
    """
    Primary Capsule Layer, implemented as paper Dynamic Routing between Capsules
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
    def __init__(self, A, B, K, stride, P):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=False)
        
        self.B = B

    def forward(self, x, sq=False):

        p = self.pose(x)
        b, c, h, w, = p.shape
        p = p.permute(0, 2, 3, 1)
        p = p.view(b, h, w, self.B, -1)
       
        if(sq):
            p = squash(p)

        a = torch.norm(p, dim=-1, keepdim=True)
        return p, a
    

class ConvCaps(nn.Module):
    """
    2D Capsule Convolutional Layer, implemented as 
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
    def __init__(self, B, C, K, stride, P):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
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
        oh = ow = int((h - K) / stride) + 1
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

    def forward(self, x, a, routing, *argv):
        """
        Forward pass
            routing: routing method
            argv[0]: number of iteration
            argv[1]: lambda for fuzzy/em routing
            argv[2]: m for fuzzy routing
        """
        assert routing in ['dynamic', 'em', 'fuzzy'], "Routing method is not implemented yet!"
        assert len(argv) > 0, "number of routing iterations need to be re-defined"
        if(routing == "fuzzy"):
            assert len(argv) == 3, "fuzzy routing need 3 arguments, 0: number of routings, 1: lambda, 2: m"
        if(routing == "em"):
            assert len(argv) == 2, "em routing need 2 arguments, 0: number of routings, 1: lambda"

        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a)
        v = self.voting(p_in)
       
        if(routing == 'em'):
            p_out, a_out = caps_EM_routing(v, a_in, self.beta_u, self.beta_a, argv[1], argv[0], g=None)
        elif(routing == 'dynamic'):
            p_out, a_out = caps_Dynamic_routing(v, a_in, argv[0], g=None)
        if(routing == 'fuzzy'):
            p_out, a_out = caps_Fuzzy_routing(v, a_in, self.beta_a, argv[1], argv[2], argv[0], g=None)
    
        
        return p_out, a_out
    

#------Shortcut Layer-----------
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
            mask = binomial.sample([N, C, 1, W, H]).to(X.device)
            return X * mask * (1.0/(1-self.p))
        return X


class CapLayer(nn.Module):
    """
    2D Capsule Convolution as conventional 3D convolution to predict higher capsules
        num_in_caps: number of capsule in L layer
        num_out_caps: number of capsule in L + 1 layer
        kernel_size: 2D kernel size
        stride: stride when doing convolutional
        in_dim: input size of a capsule
        out_dim: output size of a capsule
        groups: group convolution
    """
    def __init__(self, num_in_caps, num_out_caps, kernel_size, stride, out_dim, in_dim, groups=1, bias=True):
        super(CapLayer, self).__init__()

        self.num_out_caps = num_out_caps
        self.out_dim = out_dim
        self.num_in_caps = num_in_caps
        self.in_dim = in_dim

        assert (in_dim ** 2) % out_dim == 0
        k = in_dim ** 2 // out_dim

        self.W = nn.Conv3d(in_channels = num_in_caps, out_channels = num_out_caps * out_dim, 
            kernel_size = (k , kernel_size[0], kernel_size[1]), stride=(k, stride[0], stride[1]), groups=groups, bias=bias)

    def forward(self, x, sq=False):
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
        s = s.view(n, self.num_out_caps, self.out_dim ** 2, h, w)

        if(sq):
            s = squash(s, dim=2)
        
        return s

class LocapBlock(nn.Module):
    """
       3D depth-wise convolution to predict higher capsules
    Local Capsule Block is a computation block described in the paper "Capsule Network with Shortcut Routing"
    Input of a local block is a capsule at Lth layer
    Output of a local block is 2 capsules: pre-voting capsules (work as skip-connection), and coarse capsules (a coarse prediction for a local capsule)
    Implemented by CapLayer module, so the arguments are set according to CapLayer's arguments
    """
    def __init__(self, num_in_caps, num_out_caps, kernel_size, stride, out_dim, in_dim):
        super(LocapBlock, self).__init__()

        self.dw = CapLayer(num_in_caps, num_in_caps, kernel_size, stride, 
                            out_dim, in_dim, groups=num_in_caps, bias=False)
        self.pw = CapLayer(num_in_caps, num_out_caps, kernel_size=(1,1), stride=(1,1), 
                            out_dim=out_dim, in_dim=in_dim, bias=False)
        # self.pw = nn.Conv3d(num_in_caps, num_out_caps, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
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
 
        return prevoted_caps, coarse_caps
    
class glocapBlock(nn.Module):
    def __init__(self, num_in_caps, num_out_caps, P):
        """
        Global Capsule Block where routing algorithms will operate to calculate th final global capsule
        It can be understaned as attention from final layer to early layers
        
        """
        super(glocapBlock, self).__init__()
        self.n_classes = num_out_caps
        self.in_channels = num_in_caps
        n = self.n_classes * num_out_caps
        self.eps = 1e-06
        self.P = P
        
        self.weight = nn.Parameter(torch.randn(1, num_in_caps, 1, num_out_caps, self.P, self.P))
        self.beta_a = nn.Parameter(torch.zeros(num_out_caps, 1))
        self.beta_u = nn.Parameter(torch.zeros(num_out_caps, 1))
        
    def forward(self, l, g, routing="dynamic", *argv):
        """
        Routing
        input:
        -l: capsules at an intermediate layer (N, C_in, D, W, H)
        N: batch size, C_in: number of channels, D: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C_out, D)
        -routing: routing method
        -argv: arguments for routing
        output: 
        -g: updated global capsule
        -b: attention scores (N, C_out, C_in)
        """

        assert routing in ['dynamic', 'em', 'fuzzy'], "Routing method is not implemented yet!"
        assert len(argv) > 0, "number of routing iterations need to be re-defined"
        if(routing == "fuzzy"):
            assert len(argv) == 3, "fuzzy routing need 3 arguments, 0: number of routings, 1: lambda, 2: m"
        if(routing == "em"):
            assert len(argv) == 2, "em routing need 2 arguments, 0: number of routings, 1: lambda"
    
        #learnable transform
        N, C, D, W, H = l.size()
        l = l.permute(0, 1, 3, 4, 2).contiguous().view(N, C, W * H, self.P, self.P)
        v = l[:, :, :, None, :, :] @ self.weight
        v = v.view(N, 1, 1, self.in_channels*W*H, self.n_classes, D)

        g = g[:, None, None, None, :, :]
        
        if(routing == 'em'):
            g, a = caps_EM_routing(v, None, self.beta_u, self.beta_a, argv[1], argv[0], g=g)
        elif(routing == 'dynamic'):
            g, a = caps_Dynamic_routing(v, None, argv[0], g=g)
        if(routing == 'fuzzy'):
            g, a = caps_Fuzzy_routing(v, None, self.beta_a, argv[1], argv[2], argv[0], g=g)

        g = g.squeeze()
        a = a.squeeze()

        return a, g