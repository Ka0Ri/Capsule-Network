import torch
from torch.utils import data
import os
import cv2
path = os.getcwd() + "/21_class_new/val_5classes/"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
class MyDataset(data.Dataset):
    def __init__(self, data_files):
        self.list_names = sorted(os.listdir(data_files))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_names[index]
        # Load data and get label
        X = cv2.imread(path + "/" + ID)
        X = cv2.resize(X, (100, 100))
        y = int(ID[:2])
       
        return X, y

set_test = MyDataset(path)
loader = data.DataLoader(set_test, batch_size = 100, num_workers=8)
for batch_idx, (x, y) in enumerate(loader):
    print(batch_idx, x.size(), y.size())