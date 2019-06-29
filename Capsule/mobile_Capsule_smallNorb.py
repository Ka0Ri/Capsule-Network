"""
EM Routing Between Capsules

PyTorch implementation by Vu.
"""
import sys
sys.setrecursionlimit(1500)

import torch
import cv2
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from torch.optim import Adam, Adagrad
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm
import torchnet as tnt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import h5py

from CapsuleNet import PrimaryCaps, ConvCaps, DepthWiseConvCaps
BATCH_SIZE = 16
NUM_CLASSES = 5
NUM_EPOCHS = 250

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
path = os.getcwd()


class SmallNorb(Dataset):
    def __init__(self, name):
        hf = h5py.File(name, 'r')
        self.input_images = (np.array(hf.get('data'))).reshape((-1, 1, 96, 96))/255.0
        self.target_labels = np.array(hf.get('labels')).astype(np.long)
        hf.close()

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        return images, labels


class MobileCapsule(nn.Module):
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
    def __init__(self, A=64, B=8, C=16, D=16, E=10, K=3, P=4, iters=3):
        super(MobileCapsule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=0)
        # self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
        #                          momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.depthwise1 = DepthWiseConvCaps(B, K=7, stride=2, iters=iters)
        self.conv_caps1 = ConvCaps(B, C, K=1, P=P, stride=1, iters=iters)
        # self.conv_caps1 = ConvCaps(B, C, K=3, P=P, stride=2, iters=iters)
        self.depthwise2 = DepthWiseConvCaps(C, K=5, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K=1, P=P, stride=1, iters=iters)
        # self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)


    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        pose, a = self.primary_caps(relu1)
        pose_depthwise1, a_depthwise1 =  self.depthwise1(pose, a)
        pose1, a1 = self.conv_caps1(pose_depthwise1, a_depthwise1)
        # pose1, a1 = self.conv_caps1(pose, a)
        pose_depthwise2, a_depthwise2 =  self.depthwise2(pose1, a1)
        pose2, a2 = self.conv_caps2(pose_depthwise2, a_depthwise2)
        # pose2, a2 = self.conv_caps2(pose1, a1)
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
    model = MobileCapsule(E=NUM_CLASSES)
    model.cuda()
    capsule_loss = SpreadLoss(num_class=NUM_CLASSES)
    summary(model, input_size=(1, 96, 96))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    ##------------------init------------------------##
    optimizer = Adagrad(model.parameters())
    engine = Engine()#training loop
    
    dataset_train = SmallNorb(path + "/data/smallNorb/smallNorb_train.h5")
    dataset_test = SmallNorb(path + "/data/smallNorb/smallNorb_test.h5")
    def get_iterator(mode):
        if mode is True:
            dataset = dataset_train
        elif mode is False:
            dataset = dataset_test
        loader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=mode)

        return loader

    ##------------------log visualization------------------------##
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

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

        ground_truth = (test_sample[0])
        print(ground_truth.shape)
        # _, reconstructions = model(Variable(ground_truth).cuda())
        # reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5)))
        # reconstruction_logger.log(
        #     make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

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
        labels = torch.LongTensor(labels)
       
        # labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        data = Variable(data.float()).cuda()
        labels = Variable(labels).cuda()
        
        if training:
            classes = model(data, labels)
        else:
            classes = model(data)
        loss = capsule_loss(classes, labels, 1)

        prob = F.softmax(classes, dim = 1)
        
           
        return loss, prob


    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)