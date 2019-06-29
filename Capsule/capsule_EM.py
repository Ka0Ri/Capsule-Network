"""
EM Routing Between Capsules

PyTorch implementation by Vu.
"""
import sys
sys.setrecursionlimit(1500)

import torch

from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
import torchnet as tnt
from torchsummary import summary
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from CapsuleNet import PrimaryCaps, ConvCaps, PoolingCaps
BATCH_SIZE = 50
NUM_CLASSES = 10
NUM_EPOCHS = 50

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
    def __init__(self, A=32, B=16, C=32, D=32, E=10, K=3, P=4, iters=2):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K=3, P=4, stride=2, iters=iters, routing_mode="MS")
        self.conv_caps2 = ConvCaps(C, D, K=3, P=4, stride=1, iters=iters, routing_mode="MS")
        self.class_caps = ConvCaps(D, E, K=1, P=4, stride=1, iters=iters, coor_add=True, w_shared=True, routing_mode="MS")


    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        pose, a = self.primary_caps(relu1)
        pose1, a1 = self.conv_caps1(pose, a)
        pose2, a2 = self.conv_caps2(pose1, a1)
        pose_class, a_class = self.class_caps(pose2, a2)
        a_class = a_class.squeeze()
        pose_class = pose_class.squeeze()
   
        return a_class


class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, x, target, r):
        b, E = x.shape
        margin = self.m_min + (self.m_max - self.m_min)*r
        
        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)
        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2#minus included margin


        return loss


if __name__ == "__main__":
    r = 0
    model = CapsNet(E=NUM_CLASSES)
    model.cuda()
    capsule_loss = SpreadLoss(num_class=NUM_CLASSES)
    summary(model, input_size=(1, 28, 28))
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    ##------------------init------------------------##
    optimizer = Adam(model.parameters())
    engine = Engine()#training loop
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')#get value of attribute by key
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    ##------------------log visualization------------------------##
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        # do validation at the end of each epoch
        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.
        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
       
        #increase r each epoch
        global r
        r = r + 1./NUM_EPOCHS
        print(r)
        

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    ##------------------log visualization------------------------##

    ##------------------main flow------------------------##
    def processor(sample):
        data, labels, training = sample
        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)
        # labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()
        
        if training:
            classes = model(data, labels)
        else:
            classes = model(data)
        loss = capsule_loss(classes, labels, 1)

        prob = F.softmax(classes, dim = 1)
        return loss, prob


    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)