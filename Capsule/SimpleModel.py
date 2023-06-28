"""
Implement Models: (Convolutional Model, U-Net Model), conventional version and Capsule version. 
Authors: dtvu1707@gmail.com
"""

import torch
import torch.nn as nn
from .Layer.CapsuleLayer import ConvCaps, EffCapLayer, PrimaryCaps

# Capsule Model
def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        nn.ReLU(inplace=True),
    )

def deconvrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=1),
        nn.ReLU(inplace=True),
    )
        
class SimpleConvNet(nn.Module):
    ''' Simple (Capsule) Convolutional Neural Network 
        model_configs:
        - n_conv: number of convlutional layers
        - in, out, k, s, p: parameters of convolution
        channels in, channels out, kernel size, stride size, padding size   
    '''
    def __init__(self, model_configs):
        super(SimpleConvNet, self).__init__()

        # Parse parameters
        self.is_caps = model_configs["is_caps"]
        self.is_reconstructed = model_configs['is_reconstructed']
        self.is_eff = model_configs['is_eff']
        self.cap_dim = model_configs['caps']['cap_dims']
        self.n_caps = len(model_configs['caps']['channels']) - 1
        self.routing_config = model_configs['caps']['routing']

        # Backbone layers
        self.conv_layers = []
        conv = model_configs['convs']
        for i in range(len(conv['channels']) -1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['channels'][i], conv['channels'][i + 1], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['channels'][i + 1]),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        # Primary Capsule layer
        primarycap = model_configs['primay_caps']
        if(self.is_caps):
            self.primary_caps = PrimaryCaps(primarycap['channels'][0], primarycap['channels'][1], 
                          primarycap['k'], primarycap['s'], primarycap['p'], P=self.cap_dim)
        else:
            self.primary_caps = nn.Sequential(
                    nn.Conv2d(primarycap['channels'][0], primarycap['channels'][1], 
                            primarycap['k'], primarycap['s'], primarycap['p']),
                    nn.ReLU(),
                    nn.BatchNorm2d(primarycap['channels'][1])
                    )

        # Capsule layers
        self.caps_layers = nn.ModuleList()
        caps = model_configs['caps']
        for i in range(self.n_caps):
            if(self.is_caps):
                if(self.is_eff):
                    self.caps_layers.append(EffCapLayer(caps['channels'][i], caps['channels'][i+1], caps['k'], caps['s'], caps['p'],  
                    self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))
                else:
                    self.caps_layers.append(ConvCaps(caps['channels'][i], caps['channels'][i+1], caps['k'], caps['s'], caps['p'],  
                    self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))
            else:
                self.caps_layers.append(nn.Sequential(
                    nn.Conv2d(caps['channels'][i], caps['channels'][i + 1], caps['k'], caps['s'], caps['p']),
                    nn.ReLU(),
                    nn.BatchNorm2d(caps['channels'][i + 1]),
                ))

        # Reconstructed regularization
        if(self.is_reconstructed):
            self.n_cls = model_configs['n_cls']
            self.decoder = nn.Sequential(
                nn.Linear(self.cap_dim * self.cap_dim * self.n_cls, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                nn.Sigmoid()
            ) 

    def forward(self, x, y=None):
       
        _, _, oh, ow = x.size()
        # Convolutional layers
        x = self.conv_layers(x)
       
        # Primary Capsule
        if(self.is_caps):
            pose, a = self.primary_caps(x, sq=False)
        else:
            a = self.primary_caps(x)

        # Capsule layers
        for i in range(0, self.n_caps):
            if(self.is_caps):
                pose, a = self.caps_layers[i](pose, a)
            else: 
                a = self.caps_layers[i](a)
        
        # Reconstructions
        if(self.is_reconstructed):
            if(y == None):
                y = a.argmax(dim=-1)
            diag = torch.index_select(torch.eye(self.n_cls, device=y.device), dim=0, index=y)
            pose = (pose * diag[:, :, None]).reshape(-1, self.n_cls * self.cap_dim * self.cap_dim)
            reconstructions = self.decoder(pose)
            reconstructions = reconstructions.reshape(-1, 1, 28, 28)
            reconstructions = nn.functional.interpolate(reconstructions, size=(oh, ow))
            
            return a, reconstructions

        return a.squeeze()
    

class SimpleConvUNet(nn.Module):

    def __init__(self, model_configs):

        super(SimpleConvUNet, self).__init__()

        self.cap_dim = model_configs['caps']['cap_dims']
        self.routing_config = model_configs['caps']['routing']
        n_filters = model_configs["n_filters"]
        self.is_caps = model_configs["is_caps"]
        self.is_eff = model_configs['is_eff']

        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']

        self.conv_layers = nn.ModuleList([nn.Sequential(convrelu(model_configs["channel"], n_filters, conv["k"], 1, conv["p"]))])
        
        self.n_conv_layers = self.n_layer - 3
        for i in range(self.n_conv_layers - 1):
            self.conv_layers.append(nn.Sequential(
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv['k'], conv['s'], conv['p'])))
        
        if(self.is_caps):
            self.downsamling_layers = nn.ModuleList([PrimaryCaps(n_filters * (2 ** (self.n_conv_layers - 1)), n_filters * (2 ** self.n_conv_layers), 
                                                        conv['k'], conv['s'], conv['p'], P=self.cap_dim)])
        else:
            self.downsamling_layers = nn.ModuleList([nn.Sequential(convrelu(n_filters * (2 ** (self.n_conv_layers - 1)), n_filters * (2 ** self.n_conv_layers), 
                                                        conv["k"], conv['s'], conv["p"]), 
                                                # nn.MaxPool2d(kernel_size=2)
                                                )])
        
        for i in range(self.n_conv_layers, self.n_layer - 1):
            if(self.is_caps):
                if(self.is_eff):
                    self.downsamling_layers.append(
                    EffCapLayer(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"],
                            self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))
                else:
                    self.downsamling_layers.append(
                    ConvCaps(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"], 
                            self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))
            else:
                self.downsamling_layers.append(nn.Sequential(
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"])))

        # Expansive Path
        transpose_conv = model_configs['transpose_conv']
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_layer - 1, 1, -1):
           
            self.upsampling_layers.append(deconvrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
            self.upsampling_layers.append(convrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), conv["k"], 1, conv["p"]))

        self.upsampling_layers.append(deconvrelu(2 * n_filters, n_filters, transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
        self.upsampling_layers.append(nn.Conv2d(2 * n_filters, model_configs["channel"], conv["k"], padding='same'))

    def forward(self, x, y = None):

        encode = []

        for i in range(self.n_conv_layers):
            x = self.conv_layers[i](x)
            encode.append((None, x))
      
        a = x
        for i in range(self.n_layer - self.n_conv_layers):
            if(self.is_caps): 
                if(i == 0):
                    pose, a = self.downsamling_layers[0](a, sq=False)
                else:
                    pose, a = self.downsamling_layers[i](pose, a)
                encode.append((pose, a.squeeze()))
            else:
                a = self.downsamling_layers[i](a)
                encode.append((None, a))
           
        up = encode[-1][1]
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            c = encode[-(i//2) - 2][1]
            cat = torch.cat([up, c], dim=1)
            up = self.upsampling_layers[i+1](cat)
            
        return up