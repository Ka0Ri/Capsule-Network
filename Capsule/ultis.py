from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch import nn
from torchvision.transforms._presets import SemanticSegmentation, ObjectDetection
from functools import partial

import numpy as np
import os, glob
import PIL.Image as Image


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

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.root = data_path
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(data_path, "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(data_path, "PedMasks"))))

        n = len(imgs)
        if(mode == 'train'):
            self.imgs = imgs[:int(n*0.8)]
            self.masks = masks[:int(n*0.8)]
        elif(mode == 'val'):
            self.imgs = imgs[int(n*0.8):]
            self.masks = masks[int(n*0.8):]
        elif(mode == 'test'):
            self.imgs = imgs[int(n*0.8):]
            self.masks = masks[int(n*0.8):]

        if(self.transform is None):
            self.transform = partial(ObjectDetection)()
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        trans_img = self.transform(img)

        return trans_img, target, transforms.ToTensor()(img)
        
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

def rgb_to_2D_label(label):
 
    Land = np.array(tuple(int('#8429F6'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    Road = np.array(tuple(int('#6EC1E4'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    Vegetation = np.array(tuple(int('FEDD3A'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    Water = np.array(tuple(int('E2A929'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    Building = np.array(tuple(int('#3C1098'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    Unlabeled = np.array(tuple(int('#9B9B9B'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg [np.all(label == Building, axis = -1)] = 2
    label_seg [np.all(label == Unlabeled, axis = -1)] = 0
    label_seg [np.all(label == Land, axis = -1)] = 0
    label_seg [np.all(label == Road, axis = -1)] = 1  
    label_seg [np.all(label == Vegetation, axis = -1)] = 0   
    label_seg [np.all(label == Water, axis = -1)] = 0
   
    label_seg = label_seg[:,:,0]
    
    return label_seg
class customCUBdataset (Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.root_dir = data_path
        self.transform = transform
        self.image_dir = os.path.join(data_path,'images')
        self.image_mapping = self._load_image_mapping()
        self.label_mapping = self._load_label_mapping()
        self.bbox_mapping = self._load_bbox_mapping ()

    def _load_label_mapping (self):
        label_mapping = {}
        label_file = os.path.join(self.root_dir,'image_class_labels.txt')
        with open(label_file, 'r') as file:
            for line in file:
                image_id, label_id = line.strip().split()
                label_mapping[image_id] = int(label_id)
        return label_mapping

    def _load_image_mapping (self):
        image_mapping = {}
        image_file = os.path.join(self.root_dir,'images.txt')
        with open(image_file, 'r') as file:
            for line in file:
                image_id, image_name = line.strip().split()
                image_mapping[image_id] = image_name
        return image_mapping

    def _load_bbox_mapping (self):
        bbox_mapping = {}
        bbox_file = os.path.join(self.root_dir,'bounding_boxes.txt')
        with open(bbox_file, 'r') as file:
            for line in file:
                image_id, x, y, width, height = line.strip().split()
                bbox_mapping[image_id] = [float (x), float (y), float (width), float (height)]
        return bbox_mapping

    def __len__(self):
        return len(self.image_mapping)

    def __getitem__(self, index):
        label = self.label_mapping[str(index)]
        bbox = self.bbox_mapping[str(index)]
        image_name = self.image_mapping[str(index)]
    
        image= Image.open(os.path.join(self.image_dir,image_name)).convert('RGB')

        if self.transform:
           image = self.transform(image)

        return (image, label,bbox)


class DubaiAerialread(Dataset):
    '''
    Dubai Aerial Imagery dataset:
    https://www.kaggle.com/code/gamze1aksu/semantic-segmentation-of-aerial-imagery
    The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated 
    with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images 
    grouped into 6 larger tiles. The classes are:
    Building: #3C1098
    Land (unpaved area): #8429F6
    Road: #6EC1E4
    Vegetation: #FEDD3A
    Water: #E2A929
    Unlabeled: #9B9B9B
    '''
    def __init__(self, mode, data_path, transform=None, imgsize=224):
        input_images = []
        input_labels = []
        self.transform = transform
        for path, _, _ in os.walk(data_path):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'images':
                input_images += [os.path.join(path, file) for file in os.listdir(path)]
            if dirname == 'masks':
                input_labels += [os.path.join(path, file) for file in os.listdir(path)]
        
        img_list = sorted(input_images)
        mask_list = sorted(input_labels)

        n = len(img_list)
        if(mode == 'train'):
            self.img_list = img_list[:int(n*0.8)]
            self.mask_list = mask_list[:int(n*0.8)]
        elif(mode == 'val'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]

        if(self.transform is None):
            self.transformImg = partial(SemanticSegmentation, resize_size=(imgsize, imgsize))()
            self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = Image.open(image_path).convert('RGB')
        # load mask
        mask = Image.open(mask_path).convert('RGB')
        mask = rgb_to_2D_label(np.asarray(mask))

        if self.transform is None:
            image = self.transformImg(image)
            mask = Image.fromarray(np.uint8(mask))
            mask = self.transformAnn(mask)
        else:
            image = np.asarray(image)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.squeeze(0)
    
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
    def __init__(self, num_classes, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
       
        gt = torch.zeros_like(output, device=target.device).scatter_(1, target.unsqueeze(1), 1)
        pos_part = torch.relu(self.pos - output) ** 2
        neg_part = torch.relu(output - self.neg) ** 2
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return torch.mean(loss)
            
class SpreadLoss(nn.Module):
    '''Spread loss = |max(0, margin - (at - ai))| ^ 2'''
    def __init__(self, num_classes, m_min=0.1, m_max=1):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_classes = num_classes

    def forward(self, output, target, r=0.9):
        b = output.shape[0]
        margin = self.m_min + (self.m_max - self.m_min)*r

        gt = torch.zeros_like(output, device=target.device).scatter_(1, target.unsqueeze(1), 1)
        at = output * gt
        at_sum = torch.sum(at, dim=1, keepdim=True)
        at = (at_sum - at) * (1 - gt) + at
       
        loss = torch.mean(torch.relu(margin - (at - output)) **2)
    
        return loss