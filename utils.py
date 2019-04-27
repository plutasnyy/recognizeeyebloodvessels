import glob
import logging

from PIL import Image
from tensor import Tensor

PATCH_SIZE = 32
HALF_OF_PATCH_SIZE = int(PATCH_SIZE / 2)
SPLIT_PATCHES_SIZE = 10000
LONG_EDGE_SIZE = 700


def create_tensor_from_file():
    images_path = sorted(glob.glob('train/*'))
    for i in range(0, len(images_path), 3):
        file_name = images_path[i].replace('/', '.').split('.')[1]
        logging.info('Process filepath: {}, fileName: {}, i: {}'.format(images_path[i], file_name, i))
        base_image = Image.open(images_path[i])
        mask = Image.open(images_path[i + 1])
        vessels = Image.open(images_path[i + 2])
        yield Tensor(base_image, vessels, mask, i)
