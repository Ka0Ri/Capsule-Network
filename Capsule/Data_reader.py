import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.datasets.mnist import MNIST, FashionMNIST
import cv2
from torchvision import transforms
import random

def transform(images):
    degree = random.randint(-30, 30)
    a = random.randint(-4, 4)
    b = random.randint(-4, 4)
    R = cv2.getRotationMatrix2D((28/2,28/2), degree, 1)
    T = np.float32([[1,0,0],[0,1,0]])
    T[0, 2] = a
    T[1, 2] = b
    images = cv2.warpAffine(images,T,(28,28))
    images = cv2.warpAffine(images,R,(28,28))

    return images

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
        self.input_images = np.array(data).astype(np.float)
        self.target_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        # images = transform(images)
        return np.expand_dims(images, axis=0), labels

class FashionMnistread(Dataset):
    def __init__(self, mode):
        dataset = FashionMNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        input_images = np.array(data).astype(np.float)
        self.input_images = input_images
        # self.input_images = np.expand_dims(input_images, axis=1)
        self.target_labels = np.array(labels).astype(np.long)

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.target_labels[idx]
        # images = transform(images)
        return np.expand_dims(images, axis=0), labels