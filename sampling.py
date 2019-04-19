import logging
from random import shuffle

import numpy as np
from PIL import Image
import glob

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
from image_processor import draw_images
from tensor import Tensor
from utils import create_tensor_from_file

size = 48
half_size = int(size / 2)


def create_samples_from_tensor(tensor: Tensor):
    logging.info('Create samples from tensor: {}'.format(tensor))
    X, Y = list(), list()
    for (x, y), value in np.ndenumerate(tensor.mask):
        if x + size <= tensor.base_image.shape[0] and y + size <= tensor.base_image.shape[1]:
            center_x, center_y = x + half_size, y + half_size
            if tensor.mask[center_x][center_y] == 1:
                X.append(tensor.base_image[x: x + size, y: y + size, 0])
                Y.append(tensor.vessels[center_x][center_y])
    return X, Y


def random_undersampling(X, y):
    """
    In this moment we will lose order of samples
    """
    minority_value, majority_value = 1, 0
    new_X, new_y = list(), list()
    length = len(y)
    quantity_of_minority = sum(y)
    quantity_of_majority = length - quantity_of_minority
    indexes_list = list(range(length))
    shuffle(indexes_list)
    skipped, to_skip = 0, quantity_of_majority - quantity_of_minority
    assert to_skip >= 0
    for index in indexes_list:
        if skipped < to_skip and y[index] == majority_value:
            skipped += 1
        else:
            new_X.append(X[index])
            new_y.append(y[index])
    print("Nie zesralem sie")
    return new_X, new_y


logging.basicConfig(level=logging.INFO)
tensors_list = create_tensor_from_file(one_tensor=True)
X, y = create_samples_from_tensor(tensors_list[0])
size = int(len(X) * 0.01)
X, y = X[0:size], y[0:size]
print('Original dataset shape %s' % Counter(y))
X, y = random_undersampling(X, y)
print('Resampled dataset shape %s' % Counter(y))


