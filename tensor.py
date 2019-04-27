import logging

from numpy import asarray
from skimage import transform

from utils import LONG_EDGE_SIZE


class Tensor:
    def __init__(self, base_image, vessels, mask, id):
        assert base_image.size == vessels.size, 'Images have different sizes'
        assert mask.size == vessels.size, 'Mask has wrong size'
        self.base_image = asarray(base_image)  # 0-255
        from image_processor import correct_image
        self.corrected = correct_image(
            self.base_image)  # 0-1 TODO VERY BAD DEPENDENCY it should be moved out from this class
        self.vessels = (asarray(vessels) / 255).astype(int)  # 0-1

        if len(self.vessels.shape) == 3:
            self.vessels = self.vessels[:, :, 1]

        self.mask = (asarray(mask) / 255).astype(int)  # 0-1
        self.id = id

        self.resize_tensor(LONG_EDGE_SIZE)

    def draw_tensor(self):
        """
        TODO This is very bad dependency to other functionality, this class should be independent, it was created only for tests and it should be removed when will be unused
        """
        from image_processor import draw_images
        draw_images([self.base_image, self.corrected, self.vessels, self.mask])

    def resize_tensor(self, long_edge_size):
        long_edge = max(self.base_image.shape)
        scale = long_edge_size / long_edge
        w, h, c = self.base_image.shape
        w, h = int(w * scale), int(h * scale)

        logging.info(
            'Tensor resize from {} to {} with {} scale'.format(self.base_image.shape, (w, h, c), round(scale, 2)))

        self.base_image = transform.resize(self.base_image, (w, h, c))
        self.corrected = transform.resize(self.corrected, (w, h))
        self.vessels = transform.resize(self.vessels, (w, h))
        self.mask = transform.resize(self.mask, (w, h))

    def __repr__(self):
        return '{}: base_image: {}'.format(self.id, self.base_image.shape)
