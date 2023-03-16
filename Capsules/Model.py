import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import numpy as np
from CapsuleLayer import ConvCaps, PrimaryCaps


#-----lOSS FUNCTION------
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
            pose, a = self.caps_layers[i](pose, a, 'dynamic', 3)
           
        a = a.squeeze()
        pose = pose.squeeze()
   
        return a
    
#-----Baseline Convolutional Nerutal Network------
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


if __name__  == "__main__":
    architect_settings = {
    'n_conv': 1,
    'Conv1': {'in': 1,
              'out': 64,
              'k': 5,
              's': 2,
              'p': 0},
    'Conv2': {'in': 1,
              'out': 128,
              'k': 5,
              's': 2,
              'p': 0},
    'Conv3': {'in': 128,
              'out': 128,
              'k': 5,
              's': 2,
              'p': 0},
    'PrimayCaps': {'in': 64,
                   'out':32,
                   'k': 1,
                   's': 1,
                   'p': 0},
    'n_caps': 3,
    'Caps1': {'in': 32,
              'out': 32,
              'k': (3, 3),
              's': (2, 2)},
    'Caps2': {'in': 32,
              'out': 32,
              'k': (3, 3),
              's': (1, 1)},
    'Caps3': {'in': 32,
              'out': 10,
              'k': (3, 3),
              's': (1, 1)}
    }

    model = CapNets(input_channel=1, num_classes=2).cuda()
    input_tensor = torch.rand(2, 1, 28, 28).cuda()
    a = model(input_tensor)
    print(a)