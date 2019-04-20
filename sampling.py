import logging
import random

import numpy as np
import sklearn

from tensor import Tensor
from utils import PATCH_SIZE, HALF_OF_PATCH_SIZE


def create_samples_from_tensor(tensor: Tensor):
    #TODO instead of creating patches from all pixel, create patches only from minority class and random patches from majority class with quantity equal to minoriy class
    logging.info('Create samples from tensor: {}'.format(tensor))
    X, Y = list(), list()
    for (x, y), value in np.ndenumerate(tensor.mask):
        if x + PATCH_SIZE <= tensor.corrected.shape[0] and y + PATCH_SIZE <= tensor.corrected.shape[1]:
            center_x, center_y = x + HALF_OF_PATCH_SIZE, y + HALF_OF_PATCH_SIZE
            if tensor.mask[center_x][center_y] == 1:
                X.append(tensor.corrected[x: x + PATCH_SIZE, y: y + PATCH_SIZE])
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
    random.shuffle(indexes_list)
    skipped, to_skip = 0, quantity_of_majority - quantity_of_minority
    assert to_skip >= 0
    for index in indexes_list:
        if skipped < to_skip and y[index] == majority_value:
            skipped += 1
        else:
            new_X.append(X[index])
            new_y.append(y[index])

    result_X, result_Y = sklearn.utils.shuffle(new_X, new_y, random_state=0)
    return result_X, result_Y
