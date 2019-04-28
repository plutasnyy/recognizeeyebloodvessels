import logging
from collections import Counter
from time import strftime, gmtime

from keras.utils import to_categorical

from model import Model
from sampling import create_samples_from_tensor, random_undersampling
from utils import create_tensor_from_file, SPLIT_PATCHES_SIZE, PATCH_SIZE

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.DEBUG)
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)
# # TODO add type hints and docs for functions

model = Model()
# model.load_weights('best.hdf5')
model.compile()

for tensor in create_tensor_from_file():
    tensor.draw_tensor()
    X, y = create_samples_from_tensor(tensor)

    logging.info('Patches were created')
    print(len(X),len(y))
    logging.info('Original dataset shape {}'.format(Counter(y)))
    X, y = random_undersampling(X, y)

    logging.debug('Resampled dataset shape {}'.format(Counter(y)))

    for start_index in range(0, len(X), SPLIT_PATCHES_SIZE):
        end_index = min(start_index + SPLIT_PATCHES_SIZE, len(X))  # tricky way to avoid OutOfIndexError
        logging.info('Splitting set. Range: {}:{} Progress of this tensor: {}% Time: {}'.format(
            start_index, end_index, round(start_index / len(X) * 100), strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        X_subset, y_subset = X[start_index:end_index], y[start_index:end_index]
        logging.debug('Cut dataset result countered shape {}'.format(Counter(y_subset)))

        X_subset = np.array(X_subset).reshape(len(X_subset), PATCH_SIZE, PATCH_SIZE, 1)
        y_subset = to_categorical(y_subset)
        logging.debug('Shape X: {}, y: {}'.format(X_subset.shape, y_subset.shape))

        model.fit(X_subset, y_subset)
        break
    break
