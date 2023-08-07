"""
Implement Routing methods: Dynamic, EM, Fuzzy
Implement Capsule Layers: (PrimaryCaps, ConvCap, EfficientCaps), Capsule Head
Authors: dtvu1707@gmail.com
"""
import torch
import torch.nn as nn
from Routing import CapsuleRouting, squash, EPS
import math

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

    def __init__(self, in_channels, out_channels, n_layers=2, pooling=None):
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

class CapsulePooling(nn.Module):
    def __init__(self, size=(1, 1)) -> None:
        super().__init__()
        self.size = size
        self.pooling = nn.AdaptiveMaxPool2d(size, return_indices=True)

    def forward(self, v, a):
        b, d, p, h, w = v.shape
        a_out, indices = self.pooling(a)

        indices_tile = torch.unsqueeze(indices, 2).expand(-1, -1, p, -1, -1)
        indices_tile= indices_tile.reshape(b, d * p, -1)
        v_flatten = v.reshape(b, d * p, -1)
        v = torch.gather(v_flatten, 2, indices_tile)
        v_out = v.reshape(b, d, p, self.size[0], self.size[0])

        return v_out, a_out
    
class LinearCapsPro(nn.Module):
    def __init__(self, in_features, num_C, num_D, eps=0.0001):
        super(LinearCapsPro, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps 
        self.weight = nn.Parameter(torch.Tensor(num_C*num_D, in_features))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x, eye):
        weight_caps = self.weight[:self.num_D]
        sigma = torch.inverse(torch.mm(weight_caps, torch.t(weight_caps))+self.eps*eye)
        sigma = torch.unsqueeze(sigma, dim=0)
        for ii in range(1, self.num_C):
            weight_caps = self.weight[ii*self.num_D:(ii+1)*self.num_D]
            sigma_ = torch.inverse(torch.mm(weight_caps, torch.t(weight_caps))+self.eps*eye)
            sigma_ = torch.unsqueeze(sigma_, dim=0)
            sigma = torch.cat((sigma, sigma_))
        
        out = torch.matmul(x, torch.t(self.weight))
        out = out.view(out.shape[0], self.num_C, 1, self.num_D)
        out = torch.matmul(out, sigma)
        out = torch.matmul(out, self.weight.view(self.num_C, self.num_D, self.in_features))
        out = torch.squeeze(out, dim=2)
        out = torch.matmul(out, torch.unsqueeze(x, dim=2))
        out = torch.squeeze(out, dim=2)
        
        return torch.sqrt(out)

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
        self.P = head['caps']['cap_dims']
        self.cap_style = head['caps']['cap_style']
        self.reduce  = head['caps']['reduce']
       
        assert n_emb % (self.P * self.P) == 0, "embedding is not divisible by P * P"
        assert B % (self.P * self.P) == 0, "channel is not divisible by P * P"

        # Primary Capsule
        self.primary_capsule = nn.Sequential()
        self.primary_capsule.add_module('projection', 
                                        ProjectionHead(B, n_emb, n_layers, pooling))
        
        if(self.cap_style == 'hw'):
            self.B = B  * (n_layers == 1) + n_emb * (n_layers > 1)
            self.primary_capsule.add_module('pooling', nn.AdaptiveAvgPool2d((self.P, self.P)))
        elif(self.cap_style == 'c'):
            self.B = (B // (self.P ** 2)) * (n_layers == 1) + (n_emb // (self.P ** 2)) * (n_layers > 1) 
            if pooling == 'avg':
                self.primary_capsule.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
        elif(self.cap_style == 'ortho'):
            self.B = B  * (n_layers == 1) + n_emb * (n_layers > 1)
            self.primary_capsule.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
            self.orthogonal = LinearCapsPro(in_features=self.B, num_C=head['n_cls'], num_D=self.P*self.P)


        # Actiation for Primary Capsule
        self.activation = nn.Sequential()
        self.activation.add_module('projection', ProjectionHead(B, self.B, 2, pooling))
        if pooling == 'avg':
            self.activation.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
        self.activation.add_module('sigmoid', nn.Sigmoid())

        if self.reduce:
            self.capspooling = CapsulePooling(size=(1, 1))

        # Routing Layer
        self.routinglayer = CapsuleRouting(self.B, head['n_cls'], self.P, 
                            head['caps']['projection_type'], self.cap_style, head['caps']['routing'])
          

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
            p = p.unsqueeze(4).unsqueeze(5)
            p = p.reshape(b, d, self.P ** 2, 1, 1)
        elif(self.cap_style == 'c'):
            p = p.reshape(b, d // (self.P ** 2), self.P ** 2, h, w)
        elif(self.cap_style == 'ortho'):
            p = p.squeeze(2).squeeze(2)
            eye = torch.eye(self.P*self.P).cuda()
            a = self.orthogonal(p, eye=eye)
            return a
       
        p_out, a_out = self.routinglayer(p, a)
        if self.reduce:
            p_out, a_out = self.capspooling(p_out, a_out)
       
        a_out = torch.log(a_out / (1 - a_out + EPS))
        if get_capsules:
            return p_out, a_out
        else: 
            return a_out