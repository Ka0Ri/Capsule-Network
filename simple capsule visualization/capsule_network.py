"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
Modified by Vu
"""
import sys
sys.setrecursionlimit(1500)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torch.autograd import Variable
# from torch.optim import Adam
# from torchnet.engine import Engine
# from torchnet.logger import VisdomPlotLogger, VisdomLogger
# from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
# from tqdm import tqdm
import torchnet as tnt

BATCH_SIZE = 25
NUM_CLASSES = 10
NUM_EPOCHS = 50
NUM_ROUTING_ITERATIONS = 3

#*torch.size = size array

def augmentation(x, max_shift=2):
    _, _, height, width = x.size()
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    """
    create CapsuleLayer and routing
    """
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        """
        num_capsules = dimmension of output capsules (primary capsule)
        out_channels = dimmension of output capsules (higher level capsule)
        """
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.out_channels = out_channels

        if num_route_nodes != -1:#higher capsule layer
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:#lowest capsule layer
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, num_capsules, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(out_channels)])

    def squash(self, s, dim=-1):
        """
        Calculate non-linear squash function
        s: unormalized capsule
        v = (|s|^2)/(1+|s|^2)*(s/|s|)
        """
        squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        v = scale * s / torch.sqrt(squared_norm)
        return v

    def forward(self, x):
        """
        Routing
        x: capsules/features at l layer
        v: capsules at l + 1 layer
        """
        if self.num_route_nodes != -1:#dynamic routing
            #u = x * W matrixes in the last 2 dim are multiplied
            u = x[:,None, :, None, :] @ self.route_weights[None,: , :, :, :]
            #b = 0 => all elements of c are same, 
            #after each iteration, all elements of b in the last 2 dim are same
            b = Variable(torch.zeros(*u.size()))
            for i in range(self.num_iterations):
                c = F.softmax(b, dim=2)#calculate coefficient c = softmax(b)
                v = self.squash((c * u).sum(dim=2, keepdim=True))#non-linear activation of weighted sum v = sum(c*u)
                if i != self.num_iterations - 1:
                    b = b + (u * v).sum(dim=-1, keepdim=True)#consine similarity u*v
        else:#features -> primary capsules
            v = [capsule(x).view(x.size(0), self.num_capsules, -1).permute(0,2,1) for capsule in self.capsules]
            v = torch.cat(v, dim=1)
            v = self.squash(v)
        return v


class CapsuleNet(nn.Module):
    """
    Capsule Network: 
    Conv (256x(20x20)) -> primary Capsule ((32x6x6)x8) -> Capsule 1 (10x16) -> (Decoder) FC1 (512) -> FC2 (1024) -> FC3 (784=28x28)
    """
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # print("input", x.size())
        x = F.relu(self.conv1(x), inplace=True)
        # print("Con1", x.size())
        x = self.primary_capsules(x)
        # print("Primary capsule", x.size())
        x = self.digit_capsules(x).squeeze()
        # print("Capsule 1", x.size())
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        # print("Classes", classes.size())

        return x, classes


class CapsuleLoss(nn.Module):
    """
    Capsule Loss: 
    Loss = T*max(0, m+ - |v|)^2 + lambda*(1-T)*max(0, |v| - max-)^2 + alpha*|x-y|
    """
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)#compare the length of 2 arrays
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        
        return (margin_loss + 0.05 * reconstruction_loss) / images.size(0)

def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')#get value of attribute by key
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    

    

    
