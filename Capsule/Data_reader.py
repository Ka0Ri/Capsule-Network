import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.datasets.mnist import MNIST
import cv2


class SmallNorbread(Dataset):
    def __init__(self, name):
        hf = h5py.File(name, 'r')
        input_images = np.array(hf.get('data')).astype(np.float)
        self.input_images = np.expand_dims(input_images, axis=1)
        self.target_labels = np.array(hf.get('labels')).astype(np.long)
        hf.close()

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        return images, labels

class SVHNread(Dataset):
    def __init__(self, mode):
        dataset = SVHN(root='./data', split=mode)
        input_images = np.array(dataset.data).astype(np.float)
        self.input_images = np.expand_dims(input_images, axis=1)
        self.target_labels = np.array(dataset.labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        return images, labels

class Mnistread(Dataset):
    def __init__(self, mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        input_images = np.array(data).astype(np.float)
        self.input_images = np.expand_dims(input_images, axis=1)
        self.target_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        return images, labels