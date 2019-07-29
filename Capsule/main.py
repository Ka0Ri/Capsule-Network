import torch

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, Adagrad, Adadelta
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm
import torchnet as tnt
from torchvision import transforms
from torchsummary import summary
import os
import cv2
import random
from Data_reader import*
from Models import*
import sys

sys.setrecursionlimit(1500)
os.environ['CUDA_VISIBLE_DEVICES']= '1'
path = os.getcwd()

NUM_CLASS = 5
BATCH_SIZE = 50
EPOCHS = 400
MODEL = "Fuzzy"
DATASET = "smallNorb"
RECONSTRUCT = False

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)

def train_crop(data):
    b, d, h, w = data.shape
    temp = data.view(b, -1)
    alpha = torch.rand(b, 1) + 0.5
    beta = torch.rand(b, 1) - 0.5
    temp = alpha*temp + beta
    temp -= temp.min()
    temp /= temp.max()
    temp = temp.view(b, d, h, w)
    i = random.randint(0, 16)
    k = random.randint(0, 16)
    return temp[:, :, i:i+32, k:k+32].contiguous()

def test_crop(data):
    return data[:, :, 8:40, 8:40].contiguous()

def normalize(data):
    b, d, h, w = data.shape
    temp = data.view(b, d, -1)
    sig = temp.var(dim=-1, keepdim=True)
    m = temp.mean(dim=-1, keepdim=True)
    temp = (temp - m)/sig
    return temp.view(b, d, h, w)
    

if __name__ == "__main__":

    dataset_train = SmallNorbread(path + "/data/smallNorb/smallNorb_train48.h5")
    dataset_test = SmallNorbread(path + "/data/smallNorb/smallNorb_test32.h5")
    # dataset_train = Mnistread(True)
    # dataset_test = Mnistread(False)
    # dataset_train = SVHNread('train')
    # dataset_test = SVHNread('test')

    def get_iterator(mode):
        if mode is True:
            dataset = dataset_train
        elif mode is False:
            dataset = dataset_test
        loader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=mode)
        return loader
    
    _model = Baseline(NUM_CLASS)
    _loss = LossBaseline()
    # _model = DynamicCaps(NUM_CLASSES=NUM_CLASS, input_channel=1)
    # _loss = LossReconstruct()
    _model = EMCaps(input_channel=1, E=NUM_CLASS)
    _loss = SpreadLoss()
    
    _model.cuda()
    # summary(_model, input_size=(1, 28, 28))
    print("# parameters:", sum(param.numel() for param in _model.parameters()))
    _model.apply(init_weights)
    ##------------------init------------------------##
    log = []
    optimizer = Adam(_model.parameters(), lr=0.01)
    engine = Engine()#training loop
    meter_loss = tnt.meter.AverageValueMeter()#average over 1 epoch
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASS, normalized=True)

    ##------------------log visualization------------------------##
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASS)),
                                                     'rownames': list(range(NUM_CLASS))})
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
        info = []
       
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
        info.append(meter_loss.value()[0])
        info.append(meter_accuracy.value()[0])

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
        info.append(meter_loss.value()[0])
        info.append(meter_accuracy.value()[0])
        torch.save(_model.state_dict(), 'epochs/' + MODEL + "_" + DATASET + "_epoch.pt")
        log.append(info)


        # train_sample = next(iter(get_iterator(True)))
        # ground_truth = train_sample[0].float() / 255.0
        # ground_truth_n = train_crop(ground_truth)
        # ground_truth = ground_truth.cpu().view_as(ground_truth).data
        # ground_truth_n = ground_truth_n.cpu().view_as(ground_truth_n).data

        # ground_truth_logger.log(make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5)))
        # reconstruction_logger.log(make_grid(ground_truth_n, nrow=int(BATCH_SIZE ** 0.5)))
        # Reconstruction visualization.
        # if(RECONSTRUCT == True):
        #     test_sample = next(iter(get_iterator(False)))

        #     ground_truth = (test_sample[0].float() / 255.0)
        #     _, reconstructions = _model(Variable(ground_truth).cuda())
        #     reconstruction = reconstructions.cpu().view_as(ground_truth).data
        #     #save imgs
        #     _, _, w, h = reconstruction.shape[:]
        #     wh = int(BATCH_SIZE**0.5)
        #     imgs = np.zeros((w*wh, h*wh))
        #     k = 0
        #     for i in range(0, wh):
        #         for j in range(0, wh):
        #             imgs[(w*i):w*(i+1), (h*j):h*(j+1)] = 255*reconstruction[k]
        #             k = k + 1
        #     cv2.imwrite(path + "/imgs/" + str(state['epoch']) + ".jpg", imgs)

        #     ground_truth_logger.log(
        #     make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        #     reconstruction_logger.log(
        #     make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    ##------------------log visualization------------------------##

    ##------------------main flow------------------------##
    def processor(sample):
        data, labels, training = sample
        data = data.float() / 255.0
        labels = torch.LongTensor(labels)
        if(training):
            data = train_crop(data)
        # else:
        #     data = test_crop(data)
        data = normalize(data)
        labels = torch.eye(NUM_CLASS).index_select(dim=0, index=labels)
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()
        
        if(MODEL == "Dynamic"):
            if training:
                classes, reconstructions = _model(data, labels)
            else:
                classes, reconstructions = _model(data)
            loss = _loss(data, labels, classes, reconstructions)
        else:
            classes = _model(data)
            loss = _loss(classes, labels)
    
        return loss, classes

    engine.train(processor, get_iterator(True), maxepoch=EPOCHS, optimizer=optimizer)
    np.savetxt(MODEL + DATASET + ".csv", np.asarray(log), delimiter=",")