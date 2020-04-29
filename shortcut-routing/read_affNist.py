import scipy.io
import numpy as np
import os
from tqdm import tqdm
import struct
import h5py
import cv2
path = os.path.dirname(os.getcwd()) + "/data/affNist/"

image_size = 40 * 40
train_size = 60000
test_size = 10000

def read_data_from_file(file_index, num, train=True, one_of_n=True):
    if train:
        size = train_size
    else:
        size = test_size
    indexs = np.random.permutation(size)
    indexs = indexs[:num]
    images = read_images(file_index * size, size, train)
    labels = read_labels(file_index * size, size, train, one_of_n)

    return [images, labels]

def read_data_randomly(num, train=True, one_of_n=True):
    res_images = None
    res_labels = None
    per_num = int(num / 32)
    if train:
        size = train_size
    else:
        size = test_size
    for i in range(32):
        if i ==31:
            per_num = num - per_num * 31
        indexs = np.random.permutation(size)
        indexs = indexs[:per_num]
        images = read_images(i*size, size, train)
        labels = read_labels(i*size, size, train, one_of_n)
        # import IPython
        # IPython.embed()
        images = images[indexs]
        labels = labels[indexs]
        if res_images is None and res_labels is None:
            res_images = images
            res_labels = labels
        else:
            res_images = np.append(res_images, images)
            res_labels = np.append(res_labels, labels)
    res_images = res_images.reshape((-1, image_size))
    if one_of_n:
        res_labels = res_labels.reshape((-1, 10))
    return [res_images, res_labels]


def read_images(start, num, train=True, flat=True):
    res = None
    while num > 0:
        if train:
            file_index = int(start / 60000) + 1
            from_index = start % 60000
            data = scipy.io.loadmat(path + "training_and_validation_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= 60000:
                start += train_size - from_index # need to read from start next loop
                num = end_index - 60000 # need to read num images next loop
                end_index = 60000 #
            else:
                start += num
                num = 0
        else:
            file_index = int(start / test_size) + 1
            from_index = start % test_size
            data = scipy.io.loadmat(path + "test_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= test_size:
                start += test_size - from_index
                num = end_index - test_size
                end_index = test_size
            else:
                start += num
                num = 0
        data = data.get("affNISTdata")["image"]
        if res is None:
            res = data[0][0][:, from_index: end_index].transpose()
        else:
            res = np.append(res, data[0][0][:, from_index: end_index].transpose())
    # import IPython
    # IPython.embed()
    res = res.reshape((-1, image_size))
    return res

def read_labels(start, num, train=True, one_of_n=True):
    res = None
    while num > 0:
        if train:
            file_index = int(start / 60000) + 1
            from_index = start % 60000
            data = scipy.io.loadmat(path + "training_and_validation_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= train_size:
                start += train_size - from_index
                num = end_index - train_size
                end_index = train_size
            else:
                start += num
                num = 0
        else:
            file_index = int(start / 10000) + 1
            from_index = start % 10000
            data = scipy.io.loadmat(path + "test_batches/" + str(file_index) + ".mat")
            end_index = from_index + num
            if end_index >= test_size:
                start += test_size - from_index
                num = end_index - test_size
                end_index = test_size
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
    # import IPython
    # IPython.embed()
    if one_of_n:
        res = res.reshape(-1, 10)
    return res

def read_centered(one_of_n=True):
    data = scipy.io.loadmat(path + "training_and_validation.mat")
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

    images_translated = images
    for i in np.arange(-5, 5, 2):
        for j in np.arange(-5, 5, 2):
            print(i, j)
            for img in tqdm(images):
                M = np.float32([[1,0,i],[0,1,j]])
                dst = cv2.warpAffine(img,M,(40, 40))
                images_translated = np.append(images_translated, dst)
        

    labels = np.repeat(labels, 7, axis=0)
    
    return images_translated[1:], labels




if __name__ == "__main__":
    
    # train_data, train_label = read_data_from_file(file_index=5, num = 1)
    test_data, test_label = read_data_randomly(num=10000, one_of_n=False)
    # train_data, train_label = read_centered(one_of_n=False)
    

    # train_data = train_data.reshape((train_size, 40, 40))
    # test_data = test_data.reshape((test_size, 40, 40))

    # hf = h5py.File(path + "test_aff.h5", 'w')
    # hf.create_dataset('data', data=test_data)
    # hf.create_dataset('labels', data=test_label)
    # hf.close()

    # hf = h5py.File(path + "train_translated.h5", 'w')
    # hf.create_dataset('data', data=train_data)
    # hf.create_dataset('labels', data=train_label)
    # hf.close()
    val_data, val_label = read_data_randomly(num=10000, one_of_n=False, train=True)
    val_data = val_data.reshape((test_size, 40, 40))
    hf = h5py.File(path + "val_aff.h5", 'w')
    hf.create_dataset('data', data=val_data)
    hf.create_dataset('labels', data=val_label)
    hf.close()