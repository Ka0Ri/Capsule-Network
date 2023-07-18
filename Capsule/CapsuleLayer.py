"""
Implement Routing methods: Dynamic, EM, Fuzzy
Implement Capsule Layers: (PrimaryCaps, ConvCap, EfficientCaps), Capsule Head
Authors: dtvu1707@gmail.com
"""
import torch
import torch.nn as nn
from Routing import CapsuleRouting, squash, EPS

#--------------------Capsule Layer------------------------------------------------
    

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
    
        return self.projection(x)

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

        # Primary Capsule
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

        # Actiation for Primary Capsule
        self.activation = nn.Sequential()
        self.activation.add_module('projection', ProjectionHead(B, self.B, shuffle, 2, pooling))
        if pooling == 'avg':
            self.activation.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
        self.activation.add_module('sigmoid', nn.Sigmoid())

        # Routing Layer
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
        a_out = torch.log(a_out / (1 - a_out + EPS))
       
        if get_capsules:
            return p_out, a_out
        else: 
            return a_out