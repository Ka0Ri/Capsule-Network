import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.modules.loss import _Loss
from CapsuleNet import *


###################Base line########################
class Baseline(nn.Module):
    """  
    """
    def __init__(self, num_class=10, input_channel=1):
        super(Baseline, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=5, stride=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.1, affine=True)
        self.max1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.1, affine=True)
        self.max2 = nn.MaxPool2d(2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.conv3_bn = nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.1, affine=True)
        # self.max3 = nn.MaxPool2d(2, stride=2)
        

        self.linear1 = nn.Linear(1600, 1024)
        # self.linear2 = nn.Linear(328, 1024)
        self.drop_out = nn.Dropout(0.5)
        self.linear3 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv1_bn(x)
        x = self.max1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv2_bn(x)
        x = self.max2(x)
        # x = F.relu(self.conv3(x), inplace=True)
        # x = self.conv3_bn(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x), inplace=True)
        # x = F.relu(self.linear2(x), inplace=True)
        x = self.drop_out(x)
        x = self.linear3(x)

        return F.softmax(x, dim=1)

class LossBaseline(nn.Module):
    def __init__(self):
        super(LossBaseline, self).__init__()
        self._loss = nn.CrossEntropyLoss(size_average=True)

    def forward(self, predicted, labels):
       
        return self._loss(predicted, labels)

###################Dynamic Routing########################
class DynamicCaps(nn.Module):
    """
    """
    def __init__(self, NUM_CLASSES=10, input_channel=1):
        super(DynamicCaps, self).__init__()
        self.num_class = NUM_CLASSES
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=9, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.1, affine=True)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=2048, in_channels=8,
                                           out_channels=16)
        # self.decoder = nn.Sequential(
        #     nn.Linear(16 * NUM_CLASSES, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 32*32),
        #     nn.Sigmoid()
        # )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        # x = self.bn1(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze()
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        

        # if y is None:
        #     _, max_length_indices = classes.max(dim=1)
        #     y = Variable(torch.eye(self.num_class)).cuda().index_select(dim=0, index=max_length_indices.data)

        # reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        # return classes, reconstructions
        return classes, x.view(x.size(0), - 1)

class LossReconstruct(nn.Module):
    """
    Capsule Loss: 
    Loss = T*max(0, m+ - |v|)^2 + lambda*(1-T)*max(0, |v| - max-)^2 + alpha*|x-y|
    """
    def __init__(self):
        super(LossReconstruct, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        # assert torch.numel(images) == torch.numel(reconstructions)#compare the length of 2 arrays
        # images = images.view(reconstructions.size()[0], -1)
        # reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        
        return margin_loss / images.size(0)
        # return (margin_loss + 0.005 * reconstruction_loss) / images.size(0)


########################EM routing################################
class EMCaps(nn.Module):
    """
    """
    def __init__(self, input_channel=1, A=64, B=8, C=16, D=16, E=10, K=3, P=4, iters=2):
        super(EMCaps, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K=3, P=4, stride=2, iters=iters, routing_mode="Fuzzy")
        self.conv_caps2 = ConvCaps(C, D, K=3, P=4, stride=1, iters=iters, routing_mode="Fuzzy")
        self.class_caps = ConvCaps(C, E, K=1, P=4, stride=1, iters=iters, coor_add=True, w_shared=True, routing_mode="Fuzzy")


    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        relu1 = self.relu1(conv1)
        pose, a = self.primary_caps(relu1)
        pose1, a1 = self.conv_caps1(pose, a)
        pose2, a2 = self.conv_caps2(pose1, a1)
        pose_class, a_class = self.class_caps(pose2, a2)
        a_class = a_class.squeeze()
        pose_class = pose_class.squeeze()
   
        return a_class


class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, x, target, r=1):
        b, E = x.shape
        margin = self.m_min + (self.m_max - self.m_min)*r
        target = target.byte()
        at = torch.masked_select(x, mask=target)
        at = at.view(b, 1).repeat(1, E)
        loss = F.relu(margin - (at - x), inplace=True)
        loss = loss**2
        loss = loss.sum() / b - margin**2#minus included margin

        return loss