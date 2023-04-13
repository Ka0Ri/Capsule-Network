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
        B: number of capsule in L layer
        C: number of capsule in L + 1 layer
        K: 2D kernel size
        S: stride when doing convolutional
        P: size of a capsule
        groups: group convolution
    """
    def __init__(self, B, C, K, S, P, groups=1, bias=True):
        super(CapLayer, self).__init__()

        self.num_out_caps = C
        self.out_dim = P
        self.num_in_caps = B
        self.in_dim = P

        assert (P ** 2) % P == 0
        k = P ** 2 // P

        self.W = nn.Conv3d(in_channels = B, out_channels = C * P, 
            kernel_size = (k , K[0], K[1]), stride=(k, S[0], S[1]), groups=groups, bias=bias)

    def forward(self, x, sq=False):
        """
            Input:
                x: (n, B, P, h, w) is input capsules
                C: number of capsules, D: dimmension of a capsule
                h, w: spatial size
            return:
                v: (n, C, P, h', w') is output capsules
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
    def __init__(self, B, C, K, stride, P):
        super(LocapBlock, self).__init__()

        self.dw = CapLayer(B, B, K, stride, P, groups=B, bias=False)
        self.pw = CapLayer(B, C, K=(1,1), S=(1,1), P=P, bias=False)
        # self.pw = nn.Conv3d(num_in_caps, num_out_caps, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
        """
            Input:
                x: (n, B, P, h, w) is input capsules
                C: number of capsules, D: dimmension of a capsule
                h, w: spatial size
            return:
                v: (n, C, P, h', w') is output capsules
        """
        prevoted_caps = self.dw(x)
        coarse_caps = self.pw(prevoted_caps)
 
        return prevoted_caps, coarse_caps
    
class glocapBlock(nn.Module):
    def __init__(self, B, C, P):
        """
        Global Capsule Block where routing algorithms will operate to calculate th final global capsule
        It can be understaned as attention from final layer to early layers
        
        """
        super(glocapBlock, self).__init__()
        self.n_classes = C
        self.in_channels = B
        self.eps = 1e-06
        self.P = P
        
        self.weight = nn.Parameter(torch.randn(1, B, 1, C, self.P, self.P))
        self.beta_a = nn.Parameter(torch.zeros(C, 1))
        self.beta_u = nn.Parameter(torch.zeros(C, 1))
        
    def forward(self, l, g, routing="dynamic", *argv):
        """
        Routing
        input:
        -l: capsules at an intermediate layer (N, B, P, W, H)
        N: batch size, B: number of channels, P: capsule dim
        W, H: spatial dim
        -g: global capsule (N, C, P)
        -routing: routing method
        -argv: arguments for routing
        output: 
        -g: updated global capsule
        -a: scores (N, C)
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

        return g, a
    
class EffCapLayer(nn.Module):
    """
       Efficient depth-wise convolution to predict higher capsules
    Input of a local block is a capsule at Lth layer
    Output of a local block is a capsule at (L+1)th layer
    Implemented by CapLayer module, so the arguments are set according to CapLayer's arguments
    This Layer contains 2 operations, Depth-wise convolution acts on each Capsule Styles (channels) without routing
    and Pointwise convolution acts on each unit with routing mechanism.
        B: number of capsule in L layer
        C: number of capsule in L + 1 layer
        K: 2D kernel size
        S: stride when doing convolutional
        P: size of a capsule
        groups: group convolution
    """
    def __init__(self, B, C, K, S, P):
        super(EffCapLayer, self).__init__()

        self.P = P
        self.C = C
        self.dw = CapLayer(B, B, K, S, P, groups=B, bias=False)
        self.a_dw = nn.Sequential(
            nn.Conv2d(in_channels=B, out_channels=B, kernel_size=K, stride=S, groups=B, bias=False),
            nn.ReLU(),
            #  nn.BatchNorm2d(B)
        )
        
        self.weight = nn.Parameter(torch.randn(1, B, 1, 1, C, P, P))
        self.beta_a = nn.Parameter(torch.zeros(C, 1))
        self.beta_u = nn.Parameter(torch.zeros(C, 1))

        
    def forward(self, x, a, routing, *argv):
        """  
            Input:
                x (*, h, w, B, P*P): is input capsules
                a: activation (*, h, w, B, 1)
                B: number of capsules, P * P: dimmension of a capsule
                h, w: spatial size
                routing: routing method
                argv[0]: number of iteration
                argv[1]: lambda for fuzzy/em routing
                argv[2]: m for fuzzy routing
            return:
                p_out: (*, h', w', C, P*P) is output capsules
                a_out: (*, h', w', C,  1)
            h', w' is computed the same way as convolution layer
        """

        assert routing in ['dynamic', 'em', 'fuzzy'], "Routing method is not implemented yet!"
        assert len(argv) > 0, "number of routing iterations need to be re-defined"
        if(routing == "fuzzy"):
            assert len(argv) == 3, "fuzzy routing need 3 arguments, 0: number of routings, 1: lambda, 2: m"
        if(routing == "em"):
            assert len(argv) == 2, "em routing need 2 arguments, 0: number of routings, 1: lambda"

        x = x.permute(0, 3, 4, 1, 2)
        u = self.dw(x)
        N, B, D, W, H = u.size()
        u = u.permute(0, 1, 3, 4, 2).view(N, B, W, H, self.P, self.P)
        v = u.unsqueeze(4) @ self.weight
        v = v.view(N, W, H, B, self.C, D)
       
        a = a.squeeze().permute(0, 3, 1, 2)
        a = self.a_dw(a)
        a = a.permute(0, 2, 3, 1).unsqueeze(4)
        
      
        if(routing == 'em'):
            p_out, a_out = caps_EM_routing(v, a, self.beta_u, self.beta_a, argv[1], argv[0], g=None)
        elif(routing == 'dynamic'):
            p_out, a_out = caps_Dynamic_routing(v, a, argv[0], g=None)
        if(routing == 'fuzzy'):
            p_out, a_out = caps_Fuzzy_routing(v, a, self.beta_a, argv[1], argv[2], argv[0], g=None)
      
 
        return p_out, a_out
    
if __name__ == '__main__':

    test_layer = EffCapLayer(B = 32, C = 8, K = (5, 5), S = (2, 2), P = 4)
    p = torch.rand(2, 10, 10, 32, 16)
    a = torch.rand(2, 10, 10, 32, 1)
    p_out, a_out = test_layer(p, a, 'fuzzy', *[3, 0.01, 2])
    print(a_out.shape)
   