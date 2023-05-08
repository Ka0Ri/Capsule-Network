"""
Implement Capsule Layers: (PrimaryCaps, ConvCap, EfficientCaps), 
Authors: dtvu1707@gmail.com
"""

import torch
import torch.nn as nn
from .Routing import squash, CapsuleRouting
import numpy as np

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
        input:  (*, A, f, f)
        p: (*, B, f', f', P*P)
        a: (*, B, f', f')
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A, B, K, stride, padding, P=4):
        super(PrimaryCaps, self).__init__()

        self.pose = nn.Conv2d(in_channels=A, out_channels=B*(P*P +1),
                            kernel_size=K, stride=stride, padding=padding)

        self.B = B
        self.P = P

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if(isinstance(module, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, sq=False):

        p = self.pose(x)
        b, d, h, w =  p.shape
        p = p.reshape(b, self.B, self.P ** 2 + 1, h, w)
        p, a = torch.split(p, [self.P **2, 1], dim=2)
        if(sq):
            p = squash(p)

        a = torch.relu(a.squeeze()) 
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
        input:  (*, B, P*P, f, f) & (*, B, f, f) 
        output: (*, C, P*P, f', f') & (*, C, P*P,  f', f')
        f', f' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
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
        
        self.routinglayer = CapsuleRouting(B = self.B * self.K * self.K, C = C, P = P, mode = args[0], iters=args[1])


    def pactching(self, x):
        """
        Preparing windows for computing convolution
        Shape:
            Input:     (b, B, -1, f, f)        
            Output:    (b, B, -1, f', f', K, K)
        """
        if(len(x.shape) < 5):
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
        """
        Preparing capsule data points for routing, 
        For conv_caps:
            Input:     (b, h, w, K*K*B, P*P)
            Output:    (b, h, w, K*K*B, C, P*P)
        """
        C = self.C
        P = self.P
        b, B, psize, f, f = x.shape #b: fix dimmension, B, random dimmension
        x = x.reshape(b, B, P, P, f, f)
        v = torch.einsum('bBijhw, bBCjk -> bBCikhw', x, self.W_ij)
        v = v.reshape(b, B, C, psize, f, f)
        return v

    def forward(self, x, a):
        """
        Forward pass
        """
        # patching
        p_in, oh, ow = self.pactching(x)
        a_in, _, _ = self.pactching(a)
        a_in = a_in.squeeze(2)
        v = self.voting(p_in)

        #Routing
        p_out, a_out = self.routinglayer(v, a_in)
        
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
            N, C, P, H, W = X.shape
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = binomial.sample([N, C, 1, H, W]).to(X.device)
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
    def __init__(self, B, C, K, S=1, padding=0, P=4, groups=1, bias=False):
        super(CapLayer, self).__init__()

        self.num_out_caps = C
        self.num_in_caps = B
        self.P = P

        self.W = nn.Conv3d(in_channels = B, out_channels = C * P, 
            kernel_size = (P , K, K), stride=(P, S, S), padding=(0, padding, padding), groups=groups, bias=bias)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if(isinstance(module, nn.Conv3d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, sq=False):
        """
            Input:
                x: (n, B, P, f, f) is input capsules
                C: number of capsules, P: dimmension of a capsule
                f, f: spatial size
            return:
                v: (n, C, P, f', f') is output capsules
        """

        s = self.W(x)
        n, C, D, h, w = s.shape
        s = s.reshape(n, self.num_out_caps, self.P ** 2, h, w)
       
        if(sq):
            s = squash(s, dim=-1)
        
        return s

    
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
        S: stride when doing convolution
        padding: padding when doing convolution
        P: size of a capsule
        groups: group convolution
        weight_init: weight init method ["xavier_uniform", "xavier_normal", 
                        "kaiming_normal", "kaiming_uniform", "noisy_identity"]
        
    """
    def __init__(self, B, C, K, S=1, padding=0, P=4, weight_init='xavier_uniform', *args):
        super(EffCapLayer, self).__init__()

        self.P = P
        self.C = C
        self.B = B
        self.psize = P * P
        self.dw = CapLayer(B, B, K, S, padding, P, groups=B, bias=False)
        self.a_dw = nn.Sequential(
            nn.Conv2d(in_channels=B, out_channels=B, kernel_size=K, stride=S, padding=padding, groups=B),
            nn.ReLU(),
            # nn.BatchNorm2d(B)
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
        
        self.routinglayer = CapsuleRouting(B = B, C = C, P = P, mode = args[0], iters=args[1])
    

    def forward(self, x, a):
        """  
            Input:
                x (*, B, P, f, f): is input capsules
                a: activation (*, B, h, w)
                B: number of capsules, P * P: dimmension of a capsule
                h, w: spatial size
                routing: routing method
                argv[0]: number of iteration
                argv[1]: lambda for fuzzy/em routing
                argv[2]: m for fuzzy routing
            return:
                p_out: (*, C, P*P, f', f') is output capsules
                a_out: (*, C, f', f')
            h', w' is computed the same way as convolution layer
        """

        u = self.dw(x)
        b, B, P, h, w = u.shape
        u = u.reshape(b, B, self.P, self.P, h, w)
        v = torch.einsum('bBijhw, bBCjk -> bBCikhw', u, self.W_ij)
        v = v.reshape(b, B, self.C, P, h, w)
       
        a_in = self.a_dw(a)
        #Routing
        p_out, a_out = self.routinglayer(v, a_in)
      
        return p_out, a_out
    
    
if __name__ == '__main__':

    test_layer = EffCapLayer(32, 8, 5, 2, 2, 4, "noisy_identity", "fuzzy", 3).cuda()
    
    p = torch.rand(2, 32, 16, 40, 40).cuda()
    a = torch.rand(2, 32, 40, 40).cuda()
    p_out, a_out = test_layer(p, a)
    print(p_out.shape, a_out.shape)

    test_layer = PrimaryCaps(A = 32, B = 32, K = 3, stride = 2, padding = 1, P = 4)
    p = torch.rand(2, 32, 10, 10)
    p_out, a_out = test_layer(p, sq=True)
    print(p_out.shape, a_out.shape)
    
    test_layer = ConvCaps(32, 8, 3, 2, 1, 4, "noisy_identity", "dynamic", 3)
    p = torch.rand(2, 32, 16, 10, 10)
    a = torch.rand(2, 32, 10, 10)
    p_out, a_out = test_layer(p, a)
    print(p_out.shape, a_out.shape)

    test_layer = CapLayer(32, 8, 3, 1, 1, 4)
    p = torch.rand(2, 32, 16, 10, 10)
    p_out = test_layer(p)
    print(p_out.shape)


    print(p_out.shape, a_out.shape)

   

  