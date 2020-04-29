import numpy as np
import os
import math
import torch
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from tqdm import tqdm
from summary import *

from model_config import *
from utls import *


print(torch.__version__)




##Data Loader
if(training_settings['dataset'] == 'Mnist'):


    Train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
      
        ])
        
    Test_transform = transforms.Compose([
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])

    Train_data = Mnistread(mode='train', transform=Train_transform)
    Val_data = Mnistread(mode='val', transform=Test_transform)
    Test_data = Mnistread(mode='test', transform=Test_transform)

elif(training_settings['dataset'] == 'FMnist'):
    
    Train_transform = transforms.Compose([
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
        
    Test_transform = transforms.Compose([
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])

    Train_data = FashionMnistread(mode='train', transform=Train_transform)
    Val_data = FashionMnistread(mode='val', transform=Test_transform)
    Test_data = FashionMnistread(mode='test', transform=Test_transform)

elif(training_settings['dataset'] == 'affNist'):
    Train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])
        
    Test_transform = transforms.Compose([
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])

    Train_data = affNistread(data_path + "affNist/train_centered.h5", transform=Train_transform)
    Val_data =  affNistread(data_path + "affNist/val_aff.h5", transform=Test_transform)
    Test_data = affNistread(data_path + "affNist/test_aff.h5", transform=Test_transform)

elif(training_settings['dataset'] == 'SVHN'):

    Train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    Test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    Train_data = SVHNread(mode='train')
    Val_data = SVHNread(mode='val')
    Test_data = SVHNread(mode='test')

elif(training_settings['dataset'] == 'smallNorb'):

    Train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
        
    Test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(32),
        transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5,), std = (0.5,))
    ])

    Train_data = SmallNorbread(name=data_path + "smallNorb/smallNorb_train48.h5", transform=Train_transform)
    Val_data = SmallNorbread(name=data_path + "smallNorb/smallNorb_test48.h5", transform=Train_transform)
    Test_data = SmallNorbread(name=data_path + "smallNorb/smallNorb_test48.h5", transform=Test_transform)


Train_dataloader = DataLoader(dataset=Train_data, batch_size = training_settings['n_batch'], shuffle=True)
Val_dataloader = DataLoader(dataset=Val_data, batch_size = training_settings['n_batch'], shuffle=False)
Test_dataloader = DataLoader(dataset=Test_data, batch_size = training_settings['n_batch'], shuffle=False)

img, _ = next(iter(Test_dataloader))
img_szie = img[0].size()

if __name__ == '__main__':
    

    _model = CoreArchitect(input_channel=img_szie[0], num_classes=training_settings['n_class'])
    # _model = CapNets(input_channel=img_szie[0], num_classes=training_settings['n_class'])
    # _model = ConvNeuralNet(num_classes=training_settings['n_class'])
    _model.cuda()
    result, params_info = summary_string(_model, input_size=img_szie, device=device)
    print(result)
    optimizer = torch.optim.Adam(_model.parameters(), lr=training_settings['lr'])
    # optimizer = torch.optim.SGD(_model.parameters(), lr=training_settings['lr'])
    scheduler = StepLR(optimizer, step_size=training_settings['lr_step'], gamma=training_settings['lr_decay'])
    criterion = SpreadLoss(num_classes=training_settings['n_class'])
    # criterion = CrossEntropyLoss(num_classes=training_settings['n_class'])
    logger = Customized_Logger(model=_model, paras=training_settings, summary_string=result)

    margin = 0
    for epoch in range(1, training_settings['n_epoch'] + 1):
        print("[%d/%d]"%(epoch, training_settings['n_epoch']))
        ######Training#######
        _model.train()
        for data, classes in tqdm(Train_dataloader):
            labels = classes.cuda()
            data = data.cuda().float()

            optimizer.zero_grad()
            a = _model(data)
            loss = criterion(output=a, target=labels, r = margin)
            # loss = criterion(output=a, target=labels)
            loss.backward()
            optimizer.step()

            p_class = F.softmax(a, dim=1)
            
            logger.batch_update(p_class, labels, loss)
                      
        ######Validating#######  
        _model.eval()   
        with torch.no_grad():
            for data, classes in (Val_dataloader):

                labels = classes.cuda()
                data = data.cuda().float()
                a = _model(data)
                loss = criterion(output=a, target=labels, r = margin)
                # loss = criterion(output=a, target=labels)
                p_class = F.softmax(a, dim=1)
                
                logger.batch_update(p_class, labels, loss, train=False)
        if(epoch < 10):
                margin = margin + 0.1
        logger.epoch_update(epoch)
        scheduler.step()
    
    ###Testing###
    _model.load_state_dict(torch.load('logs/' + training_settings['log_file'] +  '.pt'))
    _model.eval()   
    with torch.no_grad():
        for data, classes in (Test_dataloader):

            labels = classes.cuda()
            data = data.cuda().float()
            a = _model(data)
            loss = criterion(output=a, target=labels, r = margin)
            # loss = criterion(output=a, target=labels)
            p_class = F.softmax(a, dim=1)
                
            logger.batch_update(p_class, labels, loss, train=False)
    logger.final_update()

            