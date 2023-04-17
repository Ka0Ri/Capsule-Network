import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import numpy as np
from CapsuleLayer import ConvCaps, PrimaryCaps, Caps_Dropout, LocapBlock, glocapBlock, EffCapLayer


#-----lOSS FUNCTION------


def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


class DiceLoss(nn.Module):

    def __init__(self, multiclass = False):
        super(DiceLoss, self).__init__()
        if(multiclass):
            self.dice_loss = multiclass_dice_coeff
        else:
            self.dice_loss = dice_coeff

        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, seg, target):
        bce = self.BCE(seg, target)
        seg = torch.sigmoid(seg)
        dice = 1 - self.dice_loss(seg, target)
        # Dice loss (objective to minimize) between 0 and 1
        return dice + bce
   

class BCE(nn.Module):
    """
    Binary Cross Entropy Loss function
    """
    def __init__(self, weight=None):
        super(BCE, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(weight)

    def forward(self, seg, labels):
        bce = self.BCE(seg, labels)
        # bce = F.binary_cross_entropy_with_logits(seg, labels)
        return bce
    
class MSE(nn.Module):
    """
    Mean Squared Error Loss function
    """
    def __init__(self, weight=None):
        super(MSE, self).__init__()
        self.MSE = nn.MSELoss(size_average=True)
        # self.BCE = nn.BCEWithLogitsLoss(weight)

    def forward(self, seg, labels):
       
        mse = self.MSE(seg, labels)
        return mse

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
        # output = F.softmax(output, dim=1)
        return self.loss(output, target)
        
class SpreadLoss(nn.Module):

    def __init__(self, num_classes, m_min=0.1, m_max=1):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_classes = num_classes

    def forward(self, output, target, r=0.9):
        b, E = output.shape
        margin = self.m_min + (self.m_max - self.m_min)*r

        gt = torch.zeros(output.size(0), self.num_classes, device=target.device)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        gt = gt.bool()
        at = torch.masked_select(output, mask=gt)
        at = at.reshape(b, 1).repeat(1, self.num_classes)

        loss = F.relu(margin - (at - output), inplace=True)
        loss = loss**2
        loss = loss.sum() / b

        return loss


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
    def __init__(self, model_configs):
        super(ConvNeuralNet, self).__init__()

        self.conv_layers = []
        for i in range(model_configs['n_conv']):
            conv = model_configs['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.caps_layers = []
        for i in range(model_configs['n_caps']):
            caps = model_configs['Caps' + str(i + 1)]
            self.caps_layers.append(nn.Sequential(
                nn.Conv2d(caps['in'], caps['out'], caps['k'], caps['s']),
                nn.ReLU(),
                nn.BatchNorm2d(caps['out']),
            ))
        self.caps_layers = nn.Sequential(*self.caps_layers)
      
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.caps_layers(out)

        return out.squeeze()

# Conventional Capsule Network
class CapNets(nn.Module):
    """
    """
    def __init__(self, model_configs):
        """
        Original Capsule Network
        - model_configs: Configurations for model
        """
        super(CapNets, self).__init__()

        self.cap_dim = model_configs['cap_dims']
        self.n_caps = model_configs['n_caps']
        self.routing_config = model_configs['routing']

        self.conv_layers = []
        for i in range(model_configs['n_conv']):
            conv = model_configs['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        primary_caps = model_configs['PrimayCaps']
        self.primary_caps = PrimaryCaps(primary_caps['in'], primary_caps['out'], 
                            primary_caps['k'], primary_caps['s'], primary_caps['p'], P=self.cap_dim)

        self.caps_layers = nn.ModuleList()
        for i in range(self.n_caps):
            caps = model_configs['Caps' + str(i + 1)]
            self.caps_layers.append(ConvCaps(caps['in'], caps['out'], caps['k'][0], caps['s'][0], self.cap_dim))

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
        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.primary_caps(x, sq=True)
        else:
            pose, a = self.primary_caps(x, sq=False)

        for i in range(0, self.n_caps):
            pose, a = self.caps_layers[i](pose, a, self.routing_config['type'], *self.routing_config['params'])
           
        a = a.squeeze()
        pose = pose.squeeze()

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
    
#-----Shortcut routing model------
class ShortcutCapsNet(nn.Module):
    """
    Shortcut Routing Network
    - model_configs: Configurations for model
    """
    def __init__(self, model_configs):
        super(ShortcutCapsNet, self).__init__()
        
        self.num_classes = model_configs['n_cls']
        self.primary_cap_num = model_configs['PrimayCaps']['out']
        self.cap_dim = model_configs['cap_dims']
        self.n_caps = model_configs['n_caps']
        self.routing_config = model_configs['routing']
        
        self.conv_layers = []
        for i in range(model_configs['n_conv']):
            conv = model_configs['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        primary_caps = model_configs['PrimayCaps']
        self.primary_caps = PrimaryCaps(primary_caps['in'], primary_caps['out'], 
                            primary_caps['k'], primary_caps['s'], primary_caps['p'], P=self.cap_dim)

        # self.dropout = Caps_Dropout(p=0.2)

        self.caps_layers = nn.ModuleList()
        self.dynamic_layers = nn.ModuleList()
        for i in range(self.n_caps):
            caps = model_configs['Caps' + str(i + 1)]
            self.caps_layers.append(LocapBlock(B=caps['in'], C=caps['out'], K=caps['k'],stride=caps['s'], padding=caps['p'], P=self.cap_dim))
            self.dynamic_layers.append(glocapBlock(B=caps['in'], C=self.num_classes, P=self.cap_dim))
            
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
        #kaiming initialization
        # self.weights_init()
   
            
    def forward(self, x, y=None):
        
        _, _, oh, ow = x.size()
        x = self.conv_layers(x)

        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.primary_caps(x, sq=True)
        else:
            pose, a = self.primary_caps(x, sq=False)
       
        p_caps = []
        for i in range(0, self.n_caps):
            p_cap, pose = self.caps_layers[i](pose)
            p_caps.append(p_cap)
            print(p_cap.shape, pose.shape)
            
        pose = pose.squeeze()
        
        for i in range(self.n_caps):
            pose, a = self.dynamic_layers[i](p_caps[i], pose, self.routing_config['type'],  *self.routing_config['params'])

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
    
    # def weights_init(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv3d, nn.Conv2d)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #             #nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, std=1e-3)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)


class EffCapNets(nn.Module):
    """
    """
    def __init__(self, model_configs):
        """
        Efficient Capsule Network
        - model_configs: Configurations for model
        """
        super(EffCapNets, self).__init__()

        self.cap_dim = model_configs['cap_dims']
        self.n_caps = model_configs['n_caps']
        self.routing_config = model_configs['routing']

        self.conv_layers = []
        for i in range(model_configs['n_conv']):
            conv = model_configs['Conv' + str(i + 1)]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv['in'], conv['out'], conv['k'], conv['s'], conv['p']),
                nn.ReLU(),
                nn.BatchNorm2d(conv['out']),
            ))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        primary_caps = model_configs['PrimayCaps']
        self.primary_caps = PrimaryCaps(primary_caps['in'], primary_caps['out'], 
                            primary_caps['k'], primary_caps['s'], primary_caps['p'], P=self.cap_dim)

        self.caps_layers = nn.ModuleList()
        for i in range(self.n_caps):
            caps = model_configs['Caps' + str(i + 1)]
            self.caps_layers.append(EffCapLayer(caps['in'], caps['out'], caps['k'], caps['s'], caps['p'], self.cap_dim))

        self.reconstructed = model_configs['reconstructed']
        if(model_configs['reconstructed']):
            self.n_cls = model_configs['n_cls']
            # self.decoder = nn.Sequential(
            #     deconvrelu(caps['out'], 128, 3, 2, padding=1),
            #     deconvrelu(128, 64, 3, 2, padding=1),
            #     nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            # )

            self.decoder = nn.Sequential(
                nn.Linear(self.cap_dim * self.cap_dim * self.n_cls, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                nn.Sigmoid()
            )
                             

    def forward(self, x, y=None):
       
        _, c, oh, ow = x.shape
        x = self.conv_layers(x)
       
        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.primary_caps(x, sq=True)
        else:
            pose, a = self.primary_caps(x, sq=False)

        for i in range(0, self.n_caps):
            pose, a = self.caps_layers[i](pose, a, self.routing_config['type'], *self.routing_config['params'])
           
        a = a.squeeze()
        pose = pose.squeeze()

        if(self.reconstructed):
            if(y == None):
                y = a.argmax(dim=-1)

            diag = torch.index_select(torch.eye(self.n_cls, device=y.device), dim=0, index=y)
            # pose = (pose * diag[:, :, None]).reshape(-1, self.n_cls, self.cap_dim, self.cap_dim)
            # reconstructions = nn.functional.interpolate(pose, size=(w // 8, h // 8))
            pose = (pose * diag[:, :, None]).reshape(-1, self.n_cls * self.cap_dim * self.cap_dim)
            reconstructions = self.decoder(pose)
            reconstructions = reconstructions.reshape(-1, 1, 28, 28)
            reconstructions = nn.functional.interpolate(reconstructions, size=(oh, ow))
            return a, reconstructions
   
        return a

# Segmentation Model

class ConvUNet(nn.Module):
    """
    Network description
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
                nn.MaxPool2d(kernel_size=2),
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
            encode.append(down)
           

        up = encode[-1]
       
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            cat = torch.cat([up, encode[-(i//2) - 2]], dim=1)
            up = self.upsampling_layers[i+1](cat)
            
        return up

class CapConvUNet(nn.Module):

    def __init__(self, model_configs):
        super(CapConvUNet, self).__init__()

        self.cap_dim = model_configs['cap_dims']
        self.routing_config = model_configs['routing']

        n_filters = model_configs["n_filters"]
        # Contracting Path
        self.n_layer = model_configs['n_layers']
        conv = model_configs['conv']

        primary_caps = model_configs['PrimayCaps']
        self.downsamling_layers = nn.ModuleList([PrimaryCaps(primary_caps['in'], n_filters, 
                                                        primary_caps['k'], (1, 1), primary_caps['p'], P=self.cap_dim)])
        for i in range(self.n_layer - 1):
            self.downsamling_layers.append(
                EffCapLayer(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), conv["k"], conv["s"], conv["p"], P=self.cap_dim)
            )


        # Expansive Path
        transpose_conv = model_configs['transpose_conv']
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_layer - 1, 1, -1):
           
            self.upsampling_layers.append(deconvrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
            self.upsampling_layers.append(convrelu(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), conv["k"], 1, conv["p"]))

        self.upsampling_layers.append(deconvrelu(2 * n_filters, n_filters, transpose_conv["k"], transpose_conv["s"], transpose_conv["p"]))
        self.upsampling_layers.append(nn.Conv2d(2 * n_filters, model_configs["channel"], conv["k"], padding='same'))

    def forward(self, x, y=None):

        _, c, w, h = x.shape
      
        if(self.routing_config['type'] == "dynamic"): 
            pose, a = self.downsamling_layers[0](x, sq=True)
        else:
            pose, a = self.downsamling_layers[0](x, sq=False)

        encode = [(pose, a.squeeze().permute(0, 3, 1, 2))]

        for i in range(1, self.n_layer):
            pose, a = self.downsamling_layers[i](pose, a, self.routing_config['type'], *self.routing_config['params'])
            encode.append((pose, a.squeeze().permute(0, 3, 1, 2)))
            print("down", pose.shape, a.shape)
           
        up = encode[-1][1]
        for i in range(0, 2 * self.n_layer - 2, 2):
            up = self.upsampling_layers[i](up)
            print("up", up.shape)
            c = encode[-(i//2) - 2][1]
            cat = torch.cat([up, c], dim=1)
            print("cat", cat.shape)
            up = self.upsampling_layers[i+1](cat)
            print("up", up.shape)
          
        return up


if __name__  == "__main__":
    # architect_settings = {
    #         "reconstructed": True,
    #         "n_cls": 10,
    #         "n_conv": 2,
    #         "Conv1": {"in": 1,
    #                 "out": 64,
    #                 "k": 5,
    #                 "s": 2,
    #                 "p": 2},
    #         "Conv2": {"in": 64,
    #                 "out": 128,
    #                 "k": 5,
    #                 "s": 2,
    #                 "p": 2},
    #         "PrimayCaps": {"in": 128,
    #                 "out": 32,
    #                 "k": 3,
    #                 "s": 2,
    #                 "p": 1},
    #         "n_caps": 2,
    #         "cap_dims": 4,
    #         "Caps1": {"in": 32,
    #                 "out": 32,
    #                 "k": [3, 3],
    #                 "s": [1, 1],
    #                 "p": [0, 0]},
    #         "Caps2": {"in": 32,
    #                 "out": 10,
    #                 "k": [3, 3],
    #                 "s": [1, 1],
    #                 "p": [0, 0]},
    #         "routing": {"type": "dynamic",
    #                 "params" : [3]}
    #         }
    # # model = EffCapNets(model_configs=architect_settings).cuda()
    # model = CapNets(model_configs=architect_settings).cuda()
    # # model = ShortcutCapsNet(architect_settings).cuda()
    # input_tensor = torch.rand(2, 1, 40, 40).cuda()
    # y = torch.tensor([2, 4]).cuda()
    # a, re = model(input_tensor, y)
    # print(re.shape)
    # print(a.shape)

    # architect_settings = {
    #     "channel": 1,
    #     "n_filters": 16,
    #     "n_layers": 6,
    #     "conv": {
    #         "k": 3,
    #         "p": 1,
    #         "s": 2
    #     },
    #     "transpose_conv": {
    #         "k": 3,
    #         "s": 2,
    #         "p": 1
    #     }
    # }
    
    # a = torch.rand(2, 1, 256, 256).cuda()
    # model = ConvUNet(model_configs=architect_settings).cuda()
    # print(model)
    # o = model(a)

    architect_settings = {
        "channel": 1,
        "n_filters": 3,
        "n_layers": 4,
        "PrimayCaps": {"in": 1,
                    "out": 16,
                    "k": 5,
                    "s": 2,
                    "p": 2},
        "cap_dims": 4,
        "conv": {
            "k": [3, 3],
            "s": [2, 2],
            "p": [1, 1]
        },
        "transpose_conv": {
            "k": 3,
            "s": 2,
            "p": 1
        },
        "routing": {"type": "dynamic",
                    "params" : [3]}
    }

    model = CapConvUNet(model_configs=architect_settings).cuda()
    a = torch.rand(2, 1, 256, 256).cuda()
    o = model(a)