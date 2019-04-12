from random import shuffle

import numpy as np
from PIL import Image
import glob

from image_processor import draw_images
from tensor import Tensor
from utils import create_tensor_from_file

size = 48
half_size = int(size / 2)


def create_samples_from_tensor(tensor: Tensor):
    X, Y = list(), list()
    for (x, y), value in np.ndenumerate(tensor.mask):
        if x + size <= tensor.base_image.shape[0] and y + size <= tensor.base_image.shape[1]:
            center_x, center_y = x + half_size, y + half_size
            if tensor.mask[center_x][center_y] == 1:
                X.append(tensor.base_image[x: x + size, y: y + size, ::])
                Y.append(tensor.vessels[center_x][center_y])
    return X, Y


tensors_list = create_tensor_from_file(one_tensor=True)
X, Y = create_samples_from_tensor(tensors_list[0])
positive = sum(Y)
all = len(Y)
negative = all - positive
balance = negative / positive
print(positive, all, negative)
print(balance)
