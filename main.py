import logging

from image_processor import draw_images
from sampling import create_samples_from_tensor, random_undersampling
from utils import create_tensor_from_file
from collections import Counter

logging.basicConfig(level=logging.INFO)
tensors_list = create_tensor_from_file(one_tensor=True)
process_only_percent_patches = 0.01  # TODO change to 1 when you will learn nn

for tensor in tensors_list:
    X, y = create_samples_from_tensor(tensor)
    size = int(len(X) * process_only_percent_patches)
    X, y = X[0:size], y[0:size]
    logging.info('Original dataset shape {}'.format(Counter(y)))
    X, y = random_undersampling(X, y)
    logging.debug('Resampled dataset shape {}'.format(Counter(y)))
    logging.debug('List sizes X: {}, y: {}'.format(len(X), len(y)))
    logging.debug('Shape X[0]: {}, y[0]: {}'.format(X[0].shape, y[0].shape))
    print(y[:16])
    draw_images(X[:16])
