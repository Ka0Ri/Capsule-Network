"""
Implement Models: (Convolutional Model, U-Net Model), conventional version and Capsule version. 
Authors: dtvu1707@gmail.com
"""

import torch
import torch.nn as nn
from .CapsuleLayer import ConvCaps, EffCapLayer, PrimaryCaps

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

#-----Baseline Convolutional Neural Network------
class ConvNeuralNet(nn.Module):
    ''' Conventional Convolutional Neural Network 
        model_configs:
        - n_conv: number of convlutional layers
        - in, out, k, s, p: parameters of convolution
        channels in, channels out, kernel size, stride size, padding size   
    '''
    def __init__(self, model_configs):
        super(ConvNeuralNet, self).__init__()

        self.conv_layers = []
        conv = model_configs['convs']
        for i in range(len(conv['channels']) -1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['channels'][i], conv['channels'][i + 1], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['channels'][i + 1]),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        primarycap = model_configs['primay_caps']
        self.primary_caps = nn.Sequential(
                nn.Conv2d(primarycap['channels'][0], primarycap['channels'][1], 
                          primarycap['k'], primarycap['s'], primarycap['p']),
                nn.ReLU(),
                nn.BatchNorm2d(primarycap['channels'][1])
                )

        self.caps_layers = []
        caps = model_configs['caps']
        for i in range(len(caps['channels']) -1):
            self.caps_layers.append(nn.Sequential(
                nn.Conv2d(caps['channels'][i], caps['channels'][i + 1], caps['k'], caps['s'], caps['p']),
                nn.ReLU(),
                nn.BatchNorm2d(caps['channels'][i + 1]),
            ))
        self.caps_layers = nn.Sequential(*self.caps_layers)
      
        
    def forward(self, x, y=None):
        out = self.conv_layers(x)
        out = self.primary_caps(out)
        out = self.caps_layers(out)

        return out.squeeze()
    
class CapNets(nn.Module):
    """
        Original Capsule Network
        - model_configs: Configurations for model
        - Model architecture: convolution -> primary capsules -> capsule layers
    """
    def __init__(self, model_configs):
        
        super(CapNets, self).__init__()

        self.cap_dim = model_configs['caps']['cap_dims']
        self.n_caps = len(model_configs['caps']['channels']) - 1
        self.routing_config = model_configs['caps']['routing']
        self.name = model_configs['name']
       
        self.conv_layers = []
        conv = model_configs['convs']
        for i in range(len(conv['channels']) -1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['channels'][i], conv['channels'][i + 1], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['channels'][i + 1]),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        primarycap = model_configs['primay_caps']
        self.primary_caps = PrimaryCaps(primarycap['channels'][0], primarycap['channels'][1], 
                          primarycap['k'], primarycap['s'], primarycap['p'], P=self.cap_dim)

        self.caps_layers = nn.ModuleList()
        caps = model_configs['caps']
        for i in range(self.n_caps):
            if("eff" in self.name):
                self.caps_layers.append(EffCapLayer(caps['channels'][i], caps['channels'][i+1], caps['k'], caps['s'], caps['p'],  
                self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))
            else:
                self.caps_layers.append(ConvCaps(caps['channels'][i], caps['channels'][i+1], caps['k'], caps['s'], caps['p'],  
                self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0]))

        self.reconstructed = model_configs['reconstructed']
        if(model_configs['reconstructed']):
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
        x = self.conv_layers(x)
        # Convolutional layers
        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.primary_caps(x, sq=True)
        else:
            pose, a = self.primary_caps(x, sq=False)

        # Capsule layers
        for i in range(0, self.n_caps):
            pose, a = self.caps_layers[i](pose, a)
        
        # Reconstructions
        if(self.reconstructed):
            if(y == None):
                y = a.argmax(dim=-1)

            diag = torch.index_select(torch.eye(self.n_cls, device=y.device), dim=0, index=y)
            pose = (pose * diag[:, :, None]).reshape(-1, self.n_cls * self.cap_dim * self.cap_dim)
            reconstructions = self.decoder(pose)
            reconstructions = reconstructions.reshape(-1, 1, 28, 28)
            reconstructions = nn.functional.interpolate(reconstructions, size=(oh, ow))
            
            return a, reconstructions

        return a
    
class ConvUNet(nn.Module):
    """
    Convolutioal UNet module
    
    """
    def __init__(self, model_configs):
        super(ConvUNet, self).__init__()

        n_filters = model_configs["n_filters"]
        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']


        self.downsamling_layers = nn.ModuleList([nn.Sequential(convrelu(model_configs["channel"], n_filters, conv["k"], 1, conv["p"]), 
                                                # nn.MaxPool2d(kernel_size=2)
                                                )])
        for i in range(self.n_layer - 1):
            self.downsamling_layers.append(nn.Sequential(
                # nn.MaxPool2d(kernel_size=2),
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"]),
                )
            )

        transpose_conv = model_configs['transpose_conv']
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_layer - 1, 1, -1):
            self.upsampling_layers.append(deconvrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
            self.upsampling_layers.append(convrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), conv["k"], 1, conv["p"]))

        # Expansive Path
        self.upsampling_layers.append(deconvrelu(2 * n_filters, n_filters, transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
        self.upsampling_layers.append(nn.Conv2d(2 * n_filters, model_configs["channel"], conv["k"], padding='same'))
        # self.out = nn.Conv2d(n_filters, model_configs["channel"], conv["k"], padding='same')
       

    def forward(self, x, y=None):

        encode = [x]
        for i in range(self.n_layer):
            down = self.downsamling_layers[i](encode[-1])
            print("down", down.shape)
            encode.append(down)

        up = encode[-1]
       
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            print("up", up.shape)
            cat = torch.cat([up, encode[-(i//2) - 2]], dim=1)
            print("cat", cat.shape)
            up = self.upsampling_layers[i+1](cat)
            print("up", up.shape)
            
        return up
    
class CapConvUNet(nn.Module):

    def __init__(self, model_configs):
        super(CapConvUNet, self).__init__()

        self.cap_dim = model_configs['caps']['cap_dims']
        self.routing_config = model_configs['caps']['routing']
      
        self.name = model_configs["name"]

        n_filters = model_configs["n_filters"]
        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']

        self.conv_layers = nn.ModuleList([nn.Sequential(convrelu(model_configs["channel"], n_filters, conv["k"], 1, conv["p"]),
                                                )])
        self.n_conv_layers = self.n_layer - 3
        for i in range(self.n_conv_layers - 1):
            self.conv_layers.append(nn.Sequential(
                convrelu(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv['k'], conv['s'], conv['p'])
            ))

        self.downsamling_layers = nn.ModuleList([PrimaryCaps(n_filters * (2 ** (self.n_conv_layers - 1)), n_filters * (2 ** self.n_conv_layers), 
                                                        conv['k'], conv['s'], conv['p'], P=self.cap_dim)])
        for i in range(self.n_conv_layers, self.n_layer - 1):
            if("eff" in self.name):
                self.downsamling_layers.append(
                EffCapLayer(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"],
                        self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0])
            )
            else:
                self.downsamling_layers.append(
                ConvCaps(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"], 
                        self.cap_dim, model_configs['caps']['init'], self.routing_config['type'], self.routing_config['params'][0])
            )


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
      
        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.downsamling_layers[0](x, sq=True)
        else:
            pose, a = self.downsamling_layers[0](x, sq=False)

        encode.append((pose, a.squeeze()))
       
        for i in range(1, self.n_layer - self.n_conv_layers):
            pose, a = self.downsamling_layers[i](pose, a)
            encode.append((pose, a.squeeze()))
           
           
        up = encode[-1][1]
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            c = encode[-(i//2) - 2][1]
            cat = torch.cat([up, c], dim=1)
            up = self.upsampling_layers[i+1](cat)
            
        return up


if __name__  == "__main__":

    architect_settings = {
            "name": "convcaps",
            "reconstructed": True,
            "n_cls": 10,
            "convs": {
                    "channels": [1, 64, 128],
                    "k": 5,
                    "s": 2,
                    "p": 2},
            "primay_caps": {
                    "channels": [128, 32],
                    "k": 3,
                    "s": 2,
                    "p": 1},
            "caps": {
                    "init": "noisy_identity",
                    "cap_dims": 4,
                    "channels": [32, 16, 10],
                    "k": 3,
                    "s": 1,
                    "p": 0,
                    "routing": {
                        "type": "dynamic",
                        "params" : [10]}}
            }
    # model = ConvNeuralNet(model_configs=architect_settings).cuda()
    model = CapNets(model_configs=architect_settings).cuda()
    input_tensor = torch.rand(2, 1, 40, 40).cuda()
    y = torch.tensor([2, 4]).cuda()
    a, re = model(input_tensor, y)
    print(a.shape)
    print(re.shape)

    # architect_settings = {
    #     "name": "capUnet",
    #     "channel": 1,
    #     "n_filters": 5,
    #     "n_layers": 4,
    #     "conv": {
    #         "k": 3,
    #         "p": 1,
    #         "s": 2
    #     },
    #     "transpose_conv": {
    #         "k": 3,
    #         "s": 2,
    #         "p": 1
    #     },
    #     "caps":{
    #             "init": "noisy_identity",
    #             "cap_dims": 4,
    #             "routing": {
    #                 "type": "dynamic",
    #                 "params" : [3]}
    #     }
    # }
    
    # a = torch.rand(2, 1, 256, 256).cuda()
    # # model = ConvUNet(model_configs=architect_settings).cuda()
    # model = CapConvUNet(model_configs=architect_settings).cuda()
    # o = model(a)
    # print(o.shape)