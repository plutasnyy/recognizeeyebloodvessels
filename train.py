import logging

from keras.utils import to_categorical

from model import Model
from sampling import create_samples_from_tensor, random_undersampling
from utils import create_tensor_from_file, PATCH_SIZE
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.INFO)
# TODO change a flow, read tensor -> learn nn, instead read all tensors -> get ot ouf memory
# TODO add type hints and docs for functions

from_ind = 0.4  # TODO It should not be a guessed percentage value, instead that we should calculate te number and split the learnin set
to_ind = 0.6
#
# model = Model()
# model.load_weights('weights/weights-improvement-49-0.90.hdf5')
# model.compile()

for tensor in create_tensor_from_file(one_tensor=False):
    pass
    # tensor.draw_tensor()
    # X, y = create_samples_from_tensor(tensor)
    # logging.info('Patches were created')
    # logging.info('Original dataset shape {}'.format(Counter(y)))
    # X, y = random_undersampling(X, y)
    #
    # logging.debug('Resampled dataset shape {}'.format(Counter(y)))
    # from_ind = int(len(X) * from_ind)
    # to_ind = int(len(X) * to_ind)
    # X, y = X[from_ind:to_ind], y[from_ind:to_ind]
    # logging.debug('Cut dataset shape {}'.format(Counter(y)))
    #
    # model.predict(X[0])
    # print(y[0])
    #
    # X = np.array(X).reshape(len(X), PATCH_SIZE, PATCH_SIZE, 1)
    # y = to_categorical(y)
    # logging.debug('Shape X: {}, y: {}'.format(X.shape, y.shape))
