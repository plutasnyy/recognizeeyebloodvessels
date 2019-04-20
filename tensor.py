from numpy import asarray

import image_processor


class Tensor:
    def __init__(self, base_image, vessels, mask, id):
        assert base_image.size == vessels.size, "Images have different sizes"
        assert mask.size == vessels.size, "Mask has wrong size"
        self.base_image = asarray(base_image)
        self.vessels = (asarray(vessels) / 255).astype(int)
        self.mask = (asarray(mask) / 255).astype(int)
        self.id = id

    def draw_tensor(self):
        """
        TODO This is very bad dependency to other functionality, this class should be independent, it was created only for tests and it should be removed when will be unused
        """
        image_processor.draw_images([self.base_image, self.vessels, self.mask])

    def __repr__(self):
        return 'base_image: {}, shape: {}, id: {}'.format(self.id, self.base_image.shape, self.id)
