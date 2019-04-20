import logging

from keras.utils import to_categorical

from model import Model
from sampling import create_samples_from_tensor, random_undersampling
from utils import create_tensor_from_file
from collections import Counter

import numpy as np

logging.basicConfig(level=logging.DEBUG)
tensors_list = create_tensor_from_file(one_tensor=True)
from_ind = 0.4 # TODO It should not be a guessed percentage value, instead that we should calculate te number and split the learnin set
to_ind = 0.6

for tensor in tensors_list:
    X, y = create_samples_from_tensor(tensor)
    logging.info('Patches were created')
    X, y = random_undersampling(X, y)
    logging.info('Original dataset shape {}'.format(Counter(y)))
    from_ind = int(len(X) * from_ind)
    to_ind = int(len(X) * to_ind)
    X, y = X[from_ind:to_ind], y[from_ind:to_ind]
    logging.debug('Resampled dataset shape {}'.format(Counter(y)))

    X = np.array(X).reshape(len(X), 32, 32, 1)
    y = to_categorical(y)
    logging.debug('Shape X: {}, y: {}'.format(X.shape, y.shape))

    model = Model()
    model.compile()
    model.fit(X, y)
