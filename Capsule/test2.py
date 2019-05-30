import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from CapsuleNet import PrimaryCaps, ConvCaps
from torchviz import make_dot
BATCH_SIZE = 50
NUM_CLASSES = 10
NUM_EPOCHS = 100

class CapsNet(nn.Module):
    """
    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        # self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
        #                          momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        # x = self.bn1(x)
        relu1 = self.relu1(conv1)
        print("conv1 ", torch.cuda.memory_allocated() / 1024**2)
        pose, a = self.primary_caps(relu1)
        print("primary caps ", torch.cuda.memory_allocated() / 1024**2)
        pose1, a1 = self.conv_caps1(pose, a)
        pose2, a2 = self.conv_caps2(pose1, a1)
        pose_class, a_class = self.class_caps(pose2, a2)
        return pose_class, a_class

if __name__ == "__main__":
    
    model = CapsNet(E=NUM_CLASSES)
    model.cuda()
    data = torch.rand(1, 1, 28, 28)
    data = Variable(data).cuda()
    model(data)

