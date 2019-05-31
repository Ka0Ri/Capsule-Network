"""
PyTorch training and visualization
Modified by Vu
"""
import sys
sys.setrecursionlimit(1500)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchsummary import summary
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.svhn import SVHN
from tqdm import tqdm
import torchnet as tnt

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 100

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
    
class Net(nn.Module):
    """
    Network description
    """
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)

    def forward(self, x):
        # print("input", x.size())
        x = F.relu(self.conv1(x), inplace=True)
        # print("Con1", x.size())
    
        return x

class LossFunction(nn.Module):
    """
    Loss function
    """
    def __init__(self):
        super(LossFunction, self).__init__()
        self.MSE = nn.MSELoss(size_average=True)

    def forward(self, labels, classes):
        loss = self.MSE(labels, classes)
        
        return loss



if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    loss_model = LossFunction()
    # model.load_state_dict(torch.load('epochs/epoch_50.pt'))
    model = model.to(device)

    summary(model, input_size=(3, 224, 224))

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
        # _, reconstructions = model(Variable(ground_truth).type(torch.FloatTensor).cuda())
        # reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5)))
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5)))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    ##------------------log visualization------------------------##

    ##------------------main flow------------------------##
    def processor(sample):
        data, labels, training = sample 
        data = augmentation(data)
        labels = torch.LongTensor(labels)
        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        data = Variable(data).to(device)
        labels = Variable(labels).to(device)

        if training:
            classes = model(data)
        else:
            classes = model(data)
        loss = loss_model(labels, classes)

        return loss, classes


    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
