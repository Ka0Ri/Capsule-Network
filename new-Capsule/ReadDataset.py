from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets import CIFAR10, SVHN
from torchvision import transforms
import torch

import numpy as np
import os, glob
import struct
import cv2
import scipy
from skimage.io import imread
from skimage.transform import resize


def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/ (xmax - xmin)

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LungCTscan(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_list = sorted(glob.glob(data_dir + '/2d_images/*.tif'))
        self.mask_list = sorted(glob.glob(data_dir + '/2d_masks/*.tif'))
        self.transform = transform
        self.image_size = 256
        if(self.transform is None):
            self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = imread(image_path)
        image = normalize_image(image)
        # resize image with 1 channel
        image = resize(image, output_shape=(self.image_size, self.image_size), preserve_range=True)

        # load image
        mask = imread(mask_path) / 255.0
        # resize mask with 1 channel
        mask = resize(mask, output_shape=(self.image_size, self.image_size), preserve_range=True)
        image, mask = np.array(image[..., np.newaxis], dtype=np.float32), np.array(mask[..., np.newaxis], dtype=np.float32)
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


class SmallNorbread(Dataset):
    def __init__(self, mode, data_path, transform=None):
        if(mode == "train"):
            input_images = self._parse_NORB_dat_file(os.path.join(data_path, "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"))
            target_labels = self._parse_NORB_cat_file(os.path.join(data_path, "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat"))
            target_labels = np.repeat(target_labels, repeats=2)
        else:
            input_images = self._parse_NORB_dat_file(os.path.join(data_path, "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat"))
            target_labels = self._parse_NORB_cat_file(os.path.join(data_path, "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat"))
            target_labels = np.repeat(target_labels, repeats=2)
        
        self.input_images = np.array(input_images).astype(np.uint8)
        self.target_labels = np.array(target_labels).astype(np.int64)
        self.transform = transform
        if(self.transform == None):
            if(mode == "train"):
                self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop((32, 32)),
                            transforms.ColorJitter(brightness=0, contrast=0),
                            transforms.ToTensor(),
                            Standardize()])
            else:
                self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.CenterCrop((32, 32)),
                            transforms.ToTensor(),
                            Standardize()
                ])

        

    def matrix_type_from_magic(self, magic_number):
        """
        Get matrix data type from magic number
        See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from small NORB files 
        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                        '1E3D4C52': 'packed matrix',
                        '1E3D4C53': 'double precision matrix',
                        '1E3D4C54': 'integer matrix',
                        '1E3D4C55': 'byte matrix',
                        '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]

    def _parse_small_NORB_header(self, file_pointer):
        """
        Parse header of small NORB binary file
            
        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file
        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': self.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    def _parse_NORB_dat_file(self, file_path):
        """
        Parse small NORB data file
        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-dat.mat` file
        Returns
        -------
        examples: ndarray
            Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
        if(os.path.isfile(file_path[:-4] + ".npy")):
            examples = np.load(file_path[:-4] + ".npy")
        else:
            with open(file_path, mode='rb') as f:

                header = self._parse_small_NORB_header(f)

                num_examples, channels, height, width = header['dimensions']

                examples = np.zeros(shape=(num_examples * channels, 48, 48), dtype=np.uint8)

                for i in range(num_examples * channels):

                    # Read raw image data and restore shape as appropriate
                    image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                    image = np.uint8(np.reshape(image, newshape=(height, width)))
                    image = cv2.resize(image, (48, 48))
                    examples[i] = image
                    #cv2.imwrite(path + "/data/smallNorb/img/" + str(i)+".jpg", examples[i])
                np.save(file_path[:-4] + ".npy", examples)
        return examples
    
    def _parse_NORB_cat_file(self, file_path):
        """
        Parse small NORB category file
        
        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-cat.mat` file
        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,) containing the category of each example
        """
        if(os.path.isfile(file_path[:-4] + ".npy")):
            examples = np.load(file_path[:-4] + ".npy")
        else:
            with open(file_path, mode='rb') as f:
                header = self._parse_small_NORB_header(f)

                num_examples, = header['dimensions']

                struct.unpack('<BBBB', f.read(4))  # ignore this integer
                struct.unpack('<BBBB', f.read(4))  # ignore this integer

                examples = np.zeros(shape=num_examples, dtype=np.int32)
            
                for i in range(num_examples):
                    category, = struct.unpack('<i', f.read(4))
                    examples[i] = category
                np.save(file_path[:-4] + ".npy", examples)
        return examples
    

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        classes = self.target_labels[idx]
        images = self.transform(images)  

        return images, classes


class affNistread(Dataset):
    def __init__(self, mode, data_path, aff=True, transform=None):
       
        self.img_size = 40 * 40
        if(aff):
            if(mode == "train"):
                self.size = 50000
                data, label = self._read_data_randomly(num=self.size, data_path=os.path.join(data_path, "training_batches"), one_of_n=False)
                data = data.reshape((self.size, 40, 40))
            else:
                self.size = 10000
                data, label = self._read_data_randomly(num=10000, data_path=os.path.join(data_path, "test_batches"), one_of_n=False)
                data = data.reshape((10000, 40, 40))
        else:
            if(mode == "train"):
                data, label = self._read_centered(data_path=os.path.join(data_path, "training"), one_of_n=False)
                data = data.reshape((50000, 40, 40))
            else:
                data, label = self._read_centered(data_path=os.path.join(data_path, "test"), one_of_n=False)
                data = data.reshape((10000, 40, 40))
        
        self.input_images = np.array(data, np.uint8)
        self.target_labels = np.array(label)
        
        self.transform = transform
        if(self.transform == None):
            self.transform = transforms.Compose([
                    # transforms.RandomCrop((32, 32), padding=0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.13066047,), (0.30810780,))
                ])


    def _read_centered(self, data_path, one_of_n=True):
        data = scipy.io.loadmat(data_path)
        images = data.get("affNISTdata")["image"]
        images = images[0][0].transpose()

        if one_of_n:
            data = data.get("affNISTdata")["label_one_of_n"]
            labels = data[0][0].transpose()
        else:
            data = data.get("affNISTdata")["label_int"]
            labels = data[0][0][0].transpose()
        
        if one_of_n:
            labels = labels.reshape(-1, 10)

        
        return images, labels
 

    def _read_images(self, start, num, data_path, flat=True):
        res = None
        while num > 0:
            file_index = int(start / self.size) + 1
            from_index = start % self.size
            data = scipy.io.loadmat(os.path.join(data_path, str(file_index) + ".mat"))
            end_index = from_index + num
            if end_index >= self.size:
                start += self.size - from_index # need to read from start next loop
                num = end_index - self.size # need to read num images next loop
                end_index = self.size #
            else:
                start += num
                num = 0
            data = data.get("affNISTdata")["image"]
            if res is None:
                res = data[0][0][:, from_index: end_index].transpose()
            else:
                res = np.append(res, data[0][0][:, from_index: end_index].transpose())
       
        res = res.reshape((-1, self.img_size))
        return res
    
    def _read_labels(self, start, num, data_path, one_of_n=True):
        res = None
        while num > 0:
            file_index = int(start / self.size) + 1
            from_index = start % self.size
            data = scipy.io.loadmat(os.path.join(data_path, str(file_index) + ".mat"))
            end_index = from_index + num
            if end_index >= self.size:
                start += self.size - from_index
                num = end_index - self.size
                end_index = self.size
            else:
                start += num
                num = 0
    
            if one_of_n:
                data = data.get("affNISTdata")["label_one_of_n"]
                if res is not None:
                    res = np.append(res, data[0][0][:, from_index: end_index].transpose())
                else:
                    res = data[0][0][:, from_index: end_index].transpose()
            else:
                data = data.get("affNISTdata")["label_int"]
                if res is not None:
                    res = np.append(res, data[0][0][0][from_index: end_index].transpose())
                else:
                    res = data[0][0][0][from_index: end_index].transpose()
        if one_of_n:
            res = res.reshape(-1, 10)
        return res
    
    def _read_data_randomly(self, num, data_path, one_of_n=True):
        res_images = None
        res_labels = None
        per_num = int(num / 32)
        size = self.size
        for i in range(32):
            if i ==31:
                per_num = num - per_num * 31
            indexs = np.random.permutation(size)
            indexs = indexs[:per_num]
            images = self._read_images(i*size, size, data_path)
            labels = self._read_labels(i*size, size, data_path, one_of_n)
            images = images[indexs]
            labels = labels[indexs]
            if res_images is None and res_labels is None:
                res_images = images
                res_labels = labels
            else:
                res_images = np.append(res_images, images)
                res_labels = np.append(res_labels, labels)
        res_images = res_images.reshape((-1, self.img_size))
        if one_of_n:
            res_labels = res_labels.reshape((-1, 10))
        return [res_images, res_labels]

    def __len__(self):
        return self.input_images.shape[0]

    def __getitem__(self, idx):
        images = self.input_images[idx]
        classes = self.target_labels[idx]
       
        images = self.transform(images)
        
        return images, classes


class Mnistread(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, transform=None):

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
        self.input_images = np.array(data)
        self.input_labels = np.array(labels).astype(np.long)
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])


    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
        images = self.transform(images)
        return images, labels
    
class SVHNread(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, transform=None):

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

        self.input_images = np.array(data, np.uint8).transpose(0, 2, 3, 1)
        self.input_labels = np.array(labels)
        if(self.transform == None):
            if(mode == "train"):
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop((32, 32)),
                    transforms.ColorJitter(brightness=0, contrast=0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                    (0.19803012, 0.20101562, 0.19703614))])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                    (0.19803012, 0.20101562, 0.19703614))])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        
        labels = self.input_labels[idx]
       
        images = self.transform(images)
        return images, labels
    

class FashionMnistread(Dataset):
    """Customized dataset loader"""
    def __init__(self, mode, data_path, transform=None):

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
        self.input_images = np.array(data, np.uint8)
        self.input_labels = np.array(labels)
        if(self.transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])

    def __len__(self):
        return (self.input_images.shape[0])

    def __getitem__(self, idx):
        images = self.input_images[idx]
        labels = self.input_labels[idx]
      
        images = self.transform(images)
        return images, labels
    
if __name__ == "__main__":
    # dataset = Mnistread(mode = "train", data_path="data")
    # dataset = SVHNread(mode = "test", data_path="data")
    # dataset = affNistread(mode = "test", data_path="data/centerMnist", aff=False)
    dataset = LungCTscan(data_dir="data/CT-scan-dataset")
    image, label = dataset[0]
    print(image.shape, label.shape)