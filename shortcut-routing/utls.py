import torch
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
import argparse
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets import CIFAR10, SVHN
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import h5py
import cv2

data_path = os.path.dirname(os.getcwd()) + "/data/"

training_settings = {
    'lr': 0.001,
    'lr_step': 20,
    'lr_decay': 0.8,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'n_epoch': 200,
    'n_batch': 64,
    'device': 'cuda:1',
    'n_class': 10,
    'log_file': 'Mnist-Fuzzy-Shortcut-large',
    'dataset': 'Mnist',
}

routing_settings = {
    'mode': 'fuzzy',
    'attention': {'n_rout' : 2},
    'fuzzy': {'n_rout': 2,
            'm' : 2,
            'lambda': 10e-1},
    'EM': {'n_rout': 2,
           'lambda': 10e-3},
    'dynamic': {'n_rout': 2}
}

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


class SmallNorbread(Dataset):
    def __init__(self, name, transform=None):
        hf = h5py.File(name, 'r')
        input_images = np.array(hf.get('data')).astype(np.uint8)
        self.input_images = input_images
        self.target_labels = np.array(hf.get('labels')).astype(np.long)

        self.transform = transform
        hf.close()

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        classes = self.target_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        images = images
        
        return images, classes

class affNistread(Dataset):
    def __init__(self, name, transform=None):
        hf = h5py.File(name, 'r')
        input_images = np.array(hf.get('data'), np.uint8)
        target_labels = np.array(hf.get('labels')).astype(np.long)
        self.input_images = input_images
        self.target_labels = target_labels
            
        self.transform = transform
        hf.close()

    def __len__(self):
        return self.input_images.shape[0]

    def __getitem__(self, idx):
        images = self.input_images[idx]
        classes = self.target_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        images = images
        
        return images, classes

class Weedread(Dataset):
    def __init__(self, name, transform=None, cl=None):
        hf = h5py.File(name, 'r')
        input_images = np.array(hf.get('data'), np.uint8)
        target_labels = np.array(hf.get('labels')).astype(np.long)
        if(cl == None):
            self.input_images = input_images
            self.target_labels = target_labels
        else:
            family_index = np.where(target_labels[:, 0] == cl)
            self.input_images = input_images[family_index]
            self.target_labels = target_labels[family_index]
            
        self.transform = transform
        hf.close()

    def __len__(self):
        return self.input_images.shape[0]

    def __getitem__(self, idx):
        images = self.input_images[idx]
        classes = self.target_labels[idx][1]
        family =  self.target_labels[idx][0]
        if self.transform is not None:
            images = self.transform(images)
        images = images
        
        return images, classes, family

class FashionMnistread(TensorDataset):
    """Customized dataset loader"""
    def __init__(self, mode, transform=None):

        if(mode == 'test'):
            dataset = FashionMNIST(root=data_path, download=True, train=False)
        else:
            dataset = FashionMNIST(root=data_path, download=True, train=True)
        data = getattr(dataset, 'data')
        labels = getattr(dataset, 'targets')
        if(mode == 'train'):
            data = data[:50000]
            labels = labels[:50000]
        elif(mode == 'val'):
            data = data[50000:60000]
            labels = labels[50000:60000]
        self.transform = transform
        self.input_images = np.array(data).astype(np.uint8)
        self.input_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels

class Mnistread(TensorDataset):
    """Customized dataset loader"""
    def __init__(self, mode, transform=None):

        if(mode == 'test'):
            dataset = MNIST(root=data_path, download=True, train=False)
        else:
            dataset = MNIST(root=data_path, download=True, train=True)
        data = getattr(dataset, 'data')
        labels = getattr(dataset, 'targets')
        if(mode == 'train'):
            data = data[:50000]
            labels = labels[:50000]
        elif(mode == 'val'):
            data = data[50000:60000]
            labels = labels[50000:60000]
        self.transform = transform
        self.input_images = np.array(data).astype(np.uint8)
        self.input_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels

class SVHNread(TensorDataset):
    """Customized dataset loader"""
    def __init__(self, mode, transform=None):

        if(mode == 'test'):
            dataset = SVHN(root=data_path, download=True, split='test')
        else:
            dataset = SVHN(root=data_path, download=True, split='train')
        data = dataset.data
        labels = dataset.labels
        if(mode == 'train'):
            data = data[:70000]
            labels = labels[:70000]
        elif(mode == 'val'):
            data = data[70000:]
            labels = labels[70000:]

        self.transform = transform
        self.input_images = np.array(data).astype(np.uint8)
        self.input_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx] / 255.0
        labels = self.input_labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels




class Customized_Logger():
    """
    logging the training process
    write to file and draw learning curves
    """
    def __init__(self, model, paras="", summary_string=""):
        
        self.file_name = paras['log_file']
        self.model = model
        num_classes = paras['n_class']

        #mesurements
        self.train_meter_loss = tnt.meter.AverageValueMeter()
        self.train_classerr = tnt.meter.ClassErrorMeter(accuracy=True)
        self.val_meter_loss = tnt.meter.AverageValueMeter()
        self.val_classerr = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
        
        #Plot Logger
        port = 8097
        self.loss_logger = VisdomPlotLogger('line', port=port, win = "Loss" + self.file_name, opts={'title': 'Loss Logger'})
        self.acc_logger = VisdomPlotLogger('line', port=port, win = "Acc" + self.file_name, opts={'title': 'Accuracy Logger'})
        self.confusion_logger = VisdomLogger('heatmap', port=port, win="confusion" + self.file_name, opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(num_classes)),
                                                                'rownames': list(range(num_classes))})
        self.reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

        #Logger   
        self.best_acc = 0
        self.best_epoch = -1
        with open('logs/' + self.file_name + '.txt', 'w+') as log_file:
            log_file.write("--SETTINGS--\n")
            if(paras != ""):
                for k in paras:
                    log_file.write('%s: %s\n'%(k, paras[k]))
            log_file.writelines(summary_string)
            log_file.write("--WRITE LOG--\n")
            log_file.write("train_acc\tval_acc\ttrain_loss\tval_loss\n")

                    
    def plot(self, epoch, recons=None):
        self.loss_logger.log(epoch, self.train_meter_loss.value()[0], name="train")
        self.acc_logger.log(epoch, self.train_classerr.value()[0], name="train")
        self.loss_logger.log(epoch, self.val_meter_loss.value()[0], name="val")
        self.acc_logger.log(epoch, self.val_classerr.value()[0], name="val")
        self.confusion_logger.log(self.confusion_meter.value())

        if(recons is not None):
            self.reconstruction_logger.log(
            make_grid(recons, nrow=int(recons.size(0) ** 0.5), padding=2, normalize=True).numpy())

    def batch_update(self, outputs, targets, loss, train=True):
        self.train_classerr.add(outputs.data, targets)
        self.train_meter_loss.add(loss.item())
        if(train == False):
            self.val_classerr.add(outputs.data, targets)
            self.val_meter_loss.add(loss.item())
            self.confusion_meter.add(outputs.data, targets)

    def epoch_update(self, epoch, recons=None, save_best = True):
        train_acc = self.train_classerr.value()[0]
        val_acc = self.val_classerr.value()[0]
        train_err = self.train_meter_loss.value()[0]
        val_err = self.val_meter_loss.value()[0]
        if(save_best == True):
            if(val_acc > self.best_acc):
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), 'logs/' + self.file_name + '.pt')
        else:
            torch.save(self.model.state_dict(), 'logs/' + self.file_name + '.pt')

        with open('logs/' + self.file_name + '.txt', 'a') as log_file:
            log_file.write('%.4f\t%.4f\t%.4f\t%.4f\n'%(train_acc, val_acc, train_err, val_err))
        
        print("training accuracy : %.4f, validation accuracy %.4f"%(train_acc, val_acc))
        self.plot(epoch, recons=recons)
        self.reset_meters()
    
    def final_update(self, training_time=0):
        test_acc = self.val_classerr.value()[0]
        test_err = self.val_meter_loss.value()[0]
        with open('logs/' + self.file_name + '.txt', 'a') as log_file:
            log_file.write('\nBest accuracy %.4f at epoch %d\n'%(self.best_acc, self.best_epoch))
            log_file.write('%.4f\t%.4f\n'%(test_acc, test_err))
            log_file.write('Training time: %.4f seconds'%(training_time))
        print("test accuracy %.4f"%(test_acc))

    def reset_meters(self):
        self.train_classerr.reset()
        self.train_meter_loss.reset()
        self.val_classerr.reset()
        self.val_meter_loss.reset()

        self.confusion_meter.reset()

class GAN_Logger():
    """
    logging the training process
    write to file and draw learning curves
    """
    def __init__(self, generator, discriminator, predictor, file_name, args="", summary_string="", num_classes=1):
        
        self.file_name = file_name
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor

        #mesurements
        self.gen_meter_loss = tnt.meter.AverageValueMeter()
        self.dis_meter_loss = tnt.meter.AverageValueMeter()
        self.pre_meter_loss = tnt.meter.AverageValueMeter()
        
        #Plot Logger
        port = 8097
        self.loss_logger = VisdomPlotLogger('line', port=port, win = "Loss", opts={'title': 'Loss Logger'})
       
        self.reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

        #Logger
        with open('logs/' + self.file_name + '.txt', 'w+') as log_file:
            log_file.write("--SETTINGS--\n")
            if(args != ""):
                for arg in vars(args):
                    log_file.write('%s: %s\n'%(arg, getattr(args, arg)))
            log_file.writelines(summary_string)
            log_file.write("--WRITE LOG--\n")
            log_file.write("train_loss\tval_loss\n")


    def plot(self, epoch, recons=None):
        self.loss_logger.log(epoch, self.gen_meter_loss.value()[0], name="generator")
        self.loss_logger.log(epoch, self.dis_meter_loss.value()[0], name="disciminator")
        self.loss_logger.log(epoch, self.pre_meter_loss.value()[0], name="predictor")

        if(recons is not None):
            #recons = (recons - np.min(recons))/(np.max(recons) - np.min(recons))
            imgs = make_grid(recons, nrow=int(recons.size(0) ** 0.5), normalize=True, range=(0, 1)).numpy()
            self.reconstruction_logger.log(imgs)
            imgs = np.transpose(imgs, (1, 2, 0))
            imgs = np.uint8(255 * imgs)
            cv2.imwrite("logs/imgs/" + str(epoch) + ".jpg", np.array(imgs))



    def batch_update(self, gen_loss, dis_loss, pre_loss, train=True):
        self.gen_meter_loss.add(gen_loss.item())
        self.dis_meter_loss.add(dis_loss.item())
        self.pre_meter_loss.add(pre_loss.item())

    def epoch_update(self, epoch, recons=None):
        gen_loss = self.gen_meter_loss.value()[0]
        dis_loss = self.dis_meter_loss.value()[0]
        pre_loss = self.pre_meter_loss.value()[0]
       
        torch.save(self.generator.state_dict(), 'logs/' + self.file_name + '_gen.pt')
        torch.save(self.discriminator.state_dict(), 'logs/' + self.file_name + '_dis.pt')
        torch.save(self.predictor.state_dict(), 'logs/' + self.file_name + '_pre.pt')

        with open('logs/' + self.file_name + '.txt', 'a') as log_file:
            log_file.write('%.4f\t%.4f\t%.4f\n'%(gen_loss, dis_loss, pre_loss))
        print("generator loss : %.4f, discriminator loss : %.4f, predictor loss  %.4f" %(gen_loss, dis_loss, pre_loss))
        self.plot(epoch, recons=recons)
        self.reset_meters()
    
    def final_update(self, training_time=0):
        with open('logs/' + self.file_name + '.txt', 'a') as log_file:
            log_file.write('Training time: %.4f seconds'%(training_time))
        self.reset_meters()

    def reset_meters(self):
        self.gen_meter_loss.reset()
        self.dis_meter_loss.reset()
        self.pre_meter_loss.reset()
        