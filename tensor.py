import logging

from PIL import Image
from numpy import asarray
from skimage import transform

import numpy as np


class Tensor:
    def __init__(self, base_image, vessels, mask, id):
        assert base_image.size == vessels.size, 'Images have different sizes'
        assert mask.size == vessels.size, 'Mask has wrong size'

        from utils import LONG_EDGE_SIZE
        long_edge = max(base_image.size)
        scale = LONG_EDGE_SIZE / long_edge
        w, h = base_image.size
        c = len(base_image.getbands())
        w, h = int(w * scale), int(h * scale)
        logging.info(
            'Tensor resize from {} to {} with {} scale'.format((base_image.size, c), (w, h, c), round(scale, 2)))

        self.base_image = asarray(base_image.resize((w, h), resample=Image.NEAREST))  # 0-255
        from image_processor import correct_image
        self.corrected = (correct_image(
            self.base_image) / 255).astype(int)  # 0-1 TODO VERY BAD DEPENDENCY it should be moved out from this class
        self.vessels = (asarray(vessels.resize((w, h), resample=Image.NEAREST)) / 255).astype(
            int)  # 0-1

        if len(self.vessels.shape) == 3:
            self.vessels = self.vessels[:, :, 1]

        self.mask = (asarray(mask.resize((w, h))) / 255).astype(int)  # 0-1
        self.id = id

    def draw_tensor(self):
        """
        TODO This is very bad dependency to other functionality, this class should be independent, it was created only for tests and it should be removed when will be unused
        """
        from image_processor import draw_images
        draw_images([self.base_image, self.corrected, self.vessels, self.mask])

    def __repr__(self):
        return '{}: base_image: {}'.format(self.id, self.base_image.shape)
