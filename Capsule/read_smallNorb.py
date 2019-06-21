import numpy as np
import os
from tqdm import tqdm
import struct
import h5py
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

        examples = np.zeros(shape=2*num_examples, dtype=np.int32)
        k = 0
        for i in tqdm(range(num_examples), desc='Loading categories...'):
            category, = struct.unpack('<i', f.read(4))
            examples[k] = category
            examples[k + 1] = category
            k = k + 2

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

        examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

        for i in tqdm(range(num_examples * channels), desc='Loading images...'):

            # Read raw image data and restore shape as appropriate
            image = struct.unpack('<' + height * width * 'B', f.read(height * width))
            image = np.uint8(np.reshape(image, newshape=(height, width)))

            examples[i] = image

    return examples

examples = _parse_NORB_dat_file(path + "/data/smallNorb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat")
labels = _parse_NORB_cat_file(path + "/data/smallNorb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat")

hf = h5py.File(path + "/data/smallNorb/smallNorb_train.h5", 'w')
hf.create_dataset('data', data=examples)
hf.create_dataset('labels', data=labels)