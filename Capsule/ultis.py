from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101, VOCSegmentation
from torchvision import transforms
import torch
from torch import nn
from torchvision.transforms._presets import SemanticSegmentation
from functools import partial
import numpy as np
import os, glob
import PIL.Image as Image
import h5py
import random


'''SEED Everything'''
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.

def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/(xmax - xmin + 10e-6)

def collate_fn(batch):
    return tuple(zip(*batch))

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LungCTscan(Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        img_list = sorted(glob.glob(data_path + '/2d_images/*.tif'))
        mask_list = sorted(glob.glob(data_path + '/2d_masks/*.tif'))
        
        n = len(img_list)
        if(mode == 'train'):
            self.img_list = img_list[:int(n*0.8)]
            self.mask_list = mask_list[:int(n*0.8)]
        elif(mode == 'val'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]
        elif(mode == 'test'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]

        self.transform = transform
        self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])
        if(self.transform is None):
            self.transformImg = partial(SemanticSegmentation, resize_size=imgsize)()
            
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = Image.open(image_path).convert('RGB')
        # resize image with 1 channel

        # load image
        mask = Image.open(mask_path).convert('L')

        if self.transform is None:
            tran_image = self.transformImg(image)
            mask = self.transformAnn(mask)
        else:
            tran_image = self.transform(image)
            mask = self.transform(mask)

        return tran_image, mask.squeeze(0), self.transformAnn(image)


        
class CIFAR10read(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, imgsize=224, transform=None):

        if(mode == 'test'):
            dataset = CIFAR10(root=data_path, download=True, train=False)
        else:
            dataset = CIFAR10(root=data_path, download=True, train=True)
        data = getattr(dataset, 'data')
        labels = getattr(dataset, 'targets')
        n = len(data)
        if(mode == 'train'):
            self.input_images = np.array(data[:int(n * 0.8)])
            self.input_labels = np.array(labels[:int(n * 0.8)])
        elif(mode == 'val'):
            self.input_images = np.array(data[int(n * 0.8):])
            self.input_labels = np.array(labels[int(n * 0.8):])
        elif(mode == 'test'):
            self.input_images = np.array(data)
            self.input_labels = np.array(labels)
    
        self.transform = transform
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
      
        trans_img = self.transform(images)
        return trans_img, labels, transforms.ToTensor()(images)
    
class CIFAR10_feats(Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):

        if(mode == 'train'):
            hf = h5py.File(data_path + '/cifar10_train.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
            hf.close()
        elif(mode == 'val'):
            hf = h5py.File(data_path + '/cifar10_val.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
        elif(mode == 'test'):
            hf = h5py.File(data_path + '/cifar10_val.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
    
        self.transform = transform
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        
        trans_img = torch.tensor(images)
        oimg = torch.sum(trans_img, dim=0, keepdim=True)

        return trans_img, labels, oimg
    
class CIFAR100_feats(Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):

        if(mode == 'train'):
            hf = h5py.File(data_path + '/cifar100_train.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
            hf.close()
        elif(mode == 'val'):
            hf = h5py.File(data_path + '/cifar100_val.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
        elif(mode == 'test'):
            hf = h5py.File(data_path + '/cifar100_val.h5', 'r')
            self.input_images = np.array(hf['data'])
            self.input_labels = np.array(hf['label'])
    
        self.transform = transform
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        
        trans_img = torch.tensor(images)
        oimg = torch.sum(trans_img, dim=0, keepdim=True)

        return trans_img, labels, oimg

class Caltech101read(Dataset):

    def __init__(self, mode, data_path, imgsize=224, transform=None):

        # load Caltech101 dataset
        full_dataset = Caltech101(root=data_path, download=True)
        train_size = int(0.8 * len(full_dataset))
        if(mode == 'train'):
            self.dataset = Subset(full_dataset, range(0, train_size))
        elif(mode == 'val'):
            self.dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
        elif(mode == 'test'):
            self.dataset = Subset(full_dataset, range(train_size, len(full_dataset)))

        self.transform = transform
        self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((imgsize, imgsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, idx):
        images = self.dataset[idx][0]
        labels = self.dataset[idx][1]
      
        trans_img = self.transform(images)
        return trans_img, labels, self.transformAnn(images)
    
class VOC2012read(Dataset):
    VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
    ]

    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    def __init__(self, mode, data_path, imgsize=224, transform=None):

        # load Caltech101 dataset
      
        if(mode == 'train'):
            self.dataset = VOCSegmentation(root=data_path, download=False, image_set='train')
        elif(mode == 'val'):
            self.dataset = VOCSegmentation(root=data_path, download=False, image_set='val')
        elif(mode == 'test'):
            self.dataset = VOCSegmentation(root=data_path, download=False, image_set='val')

        self.transform = transform
        self.transformOIMG = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])
        if(self.transform is None):
            self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    ])
            self.transformImg = partial(SemanticSegmentation, resize_size=(imgsize, imgsize))()

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        mask = np.array(mask)
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC2012read.VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC2012read.VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
       
        segmentation_mask = np.argmax(segmentation_mask, axis=-1).astype(np.uint8)
        # print(np.sum(segmentation_mask))
        segmentation_mask = Image.fromarray(segmentation_mask)
        return segmentation_mask
    
    def __len__(self):
        return (len(self.dataset))
    
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        mask = self.dataset[idx][1].convert("RGB")
        mask = self._convert_to_segmentation_mask(mask)
      
        if self.transform is None:
            tran_image = self.transformImg(image)
            mask = self.transformAnn(mask)
            mask = torch.tensor(np.array(mask))
        else:
            tran_image = self.transform(image=image, mask=mask)
            image = tran_image["image"]
            mask = tran_image["mask"]

        return tran_image, mask, self.transformOIMG(image)



class CIFAR100read(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, imgsize=224, transform=None):

        # load CIFAR100 datase
        if(mode == 'test'):
            dataset = CIFAR100(root=data_path, download=True, train=False)
        else:
            dataset = CIFAR100(root=data_path, download=True, train=True)
        data = getattr(dataset, 'data')
        labels = getattr(dataset, 'targets')
        n = len(data)
        if(mode == 'train'):
            self.input_images = np.array(data[:int(n * 0.8)])
            self.input_labels = np.array(labels[:int(n * 0.8)])
        elif(mode == 'val'):
            self.input_images = np.array(data[int(n * 0.8):])
            self.input_labels = np.array(labels[int(n * 0.8):])
        elif(mode == 'test'):
            self.input_images = np.array(data)
            self.input_labels = np.array(labels)
    
        self.transform = transform
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(imgsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
      
        trans_img = self.transform(images)
        return trans_img, labels, transforms.ToTensor()(images)

#---------------------------------------lOSS FUNCTION-----------------------------------------------

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


class DiceLoss(nn.Module):

    def __init__(self, multiclass = False):
        super(DiceLoss, self).__init__()
        if(multiclass):
            self.dice_loss = multiclass_dice_coeff
        else:
            self.dice_loss = dice_coeff

        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, seg, target):
        bce = self.BCE(seg, target)
        seg = torch.sigmoid(seg)
        dice = 1 - self.dice_loss(seg, target)
        # Dice loss (objective to minimize) between 0 and 1
        return dice + bce
       
class MarginLoss(nn.Module):
    """
    Loss = T*max(0, m+ - |v|)^2 + lambda*(1-T)*max(0, |v| - max-)^2 + alpha*|x-y|
    """
    def __init__(self, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
       
        gt = torch.zeros_like(output, device=target.device).scatter_(1, target.unsqueeze(1), 1)
        pos_part = torch.relu(self.pos - output) ** 2
        neg_part = torch.relu(output - self.neg) ** 2
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return torch.sum(loss)
            
class SpreadLoss(nn.Module):
    '''Spread loss = |max(0, margin - (at - ai))| ^ 2'''
    def __init__(self, m_min=0.1, m_max=1):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max

    def forward(self, output, target, r=0.9):
        b = output.shape[0]
        margin = self.m_min + (self.m_max - self.m_min)*r

        gt = torch.zeros_like(output, device=target.device).scatter_(1, target.unsqueeze(1), 1)
        at = output * gt
        at_sum = torch.sum(at, dim=1, keepdim=True)
        at = (at_sum - at) * (1 - gt) + at
       
        loss = torch.sum(torch.relu(margin - (at - output)) **2)
    
        return loss