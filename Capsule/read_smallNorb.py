import numpy as np
import os
from tqdm import tqdm
import struct
import h5py
import cv2
path = os.getcwd()

def matrix_type_from_magic(magic_number):
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

def _parse_small_NORB_header(file_pointer):
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
                        'matrix_type': matrix_type_from_magic(magic),
                        'dimensions': dimensions}
    return file_header_data


def _parse_NORB_cat_file(file_path):
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
    with open(file_path, mode='rb') as f:
        header = _parse_small_NORB_header(f)

        num_examples, = header['dimensions']

        struct.unpack('<BBBB', f.read(4))  # ignore this integer
        struct.unpack('<BBBB', f.read(4))  # ignore this integer

        examples = np.zeros(shape=num_examples, dtype=np.int32)
      
        for i in tqdm(range(num_examples), desc='Loading categories...'):
            category, = struct.unpack('<i', f.read(4))
            examples[i] = category

        return examples

  
def _parse_NORB_dat_file(file_path):
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
    with open(file_path, mode='rb') as f:

        header = _parse_small_NORB_header(f)

        num_examples, channels, height, width = header['dimensions']

        examples = np.zeros(shape=(num_examples * channels, 48, 48), dtype=np.uint8)

        for i in tqdm(range(num_examples * channels), desc='Loading images...'):

            # Read raw image data and restore shape as appropriate
            image = struct.unpack('<' + height * width * 'B', f.read(height * width))
            image = np.uint8(np.reshape(image, newshape=(height, width)))
            image = cv2.resize(image, (48, 48))
            examples[i] = image
            cv2.imwrite(path + "/data/smallNorb/img/" + str(i)+".jpg", examples[i])

    return examples

# index = range(0, 46000, 2)
# examples1 = _parse_NORB_dat_file(path + "/data/smallNorb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat")
# labels1 = _parse_NORB_cat_file(path + "/data/smallNorb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat")
# labels1 = np.repeat(labels1, 2)
# examples1 = examples1[index]
# labels1 = labels1[index]

# examples2 = _parse_NORB_dat_file(path + "/data/smallNorb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat")
# labels2 = _parse_NORB_cat_file(path + "/data/smallNorb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat")
# labels2 = np.repeat(labels2, 2)
# examples2 = examples2[index]
# labels2 = labels2[index]

# data = np.concatenate([examples1, examples2], axis=0)
# labels = np.concatenate([labels1, labels2], axis=0)

# index_train = range(0, 46000, 2)
# index_test = range(1, 46000, 2)
# train_data = data[index_train]
# train_labels = labels[index_train]
# test_data = data[index_test]
# test_labels = labels[index_test]

# hf = h5py.File(path + "/data/smallNorb/smallNorb_ctest32.h5", 'w')
# hf.create_dataset('data', data=test_data)
# hf.create_dataset('labels', data=test_labels)
# hf.close()

# hf = h5py.File(path + "/data/smallNorb/smallNorb_ctrain32.h5", 'w')
# hf.create_dataset('data', data=train_data)
# hf.create_dataset('labels', data=train_labels)
# hf.close()