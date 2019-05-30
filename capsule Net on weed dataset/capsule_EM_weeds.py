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
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
import torchnet as tnt

from CapsuleNet import PrimaryCaps, ConvCaps

BATCH_SIZE = 50
NUM_CLASSES = 21
NUM_EPOCHS = 100
NUM_ROUTING_ITERATIONS = 3
DATA_SIZE = 64

#*torch.size = size array

class MyDataset(data.Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.list_names = sorted(os.listdir(data_files))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_names[index]
        # Load data and get label
        X = cv2.imread(self.data_files + "/" + ID)
        X = cv2.resize(X, (DATA_SIZE, DATA_SIZE))/255.0
        X = np.transpose(X, (2, 1, 0))
        X = np.array(X, dtype=np.float32)
        y = int(ID[:2]) - 1
       
        return X, y


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
    def __init__(self, A=32, B=8, C=16, D=16, E=10, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=A,
                               kernel_size=5, stride=2)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=2, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=2, iters=iters, coor_add=True, w_shared=True)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3*64*64),
            nn.Sigmoid()
        )


    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        pose, a = self.primary_caps(relu1)
        pose1, a1 = self.conv_caps1(pose, a)
        # pose2, a2 = self.conv_caps2(pose1, a1)
        pose_class, a_class = self.class_caps(pose1, a1)
        a_class = a_class.squeeze()
        pose_class = pose_class.squeeze()
   
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = a_class.max(dim=1)
            select = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
        else:
            select = torch.eye(NUM_CLASSES).cuda().index_select(dim=0, index=y)
        reconstructions = self.decoder((pose_class * select[:, :, None]).view(x.size(0), -1))
        return a_class, reconstructions


class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class
        self.reconstruction_loss = nn.MSELoss(size_average=True)

    def forward(self, images, x, target, reconstructions, r):
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

        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return loss +  0.005 * reconstruction_loss


if __name__ == "__main__":
    r = 0
    model = CapsNet(E=NUM_CLASSES)
    model.cuda()
    capsule_loss = SpreadLoss(num_class=NUM_CLASSES)
  
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    ##------------------init------------------------##
    optimizer = Adam(model.parameters())
    engine = Engine()#training loop
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    def get_iterator(mode):
        if mode is True:
            path = os.getcwd() + "/21_class_new/train_all_classes/"
        elif mode is False:
            path = os.getcwd() + "/21_class_new/val_all_classes/"
        set_data = MyDataset(path)
        loader = data.DataLoader(set_data, batch_size = BATCH_SIZE, num_workers=8, shuffle=True)
        return loader

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

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % 0)

        # Reconstruction visualization.
        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0])
        _, reconstructions = model(Variable(ground_truth).type(torch.FloatTensor).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5)))
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5)))

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
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()
        
        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)
        loss = capsule_loss(data, classes, labels, reconstructions, r)

        prob = F.softmax(classes, dim = 1)
        return loss, prob


    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
