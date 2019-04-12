from numpy import asarray


class Tensor:
    def __init__(self, base_image, vessels, mask):
        assert base_image.size == vessels.size, "Images have different sizes"
        assert mask.size == vessels.size, "Mask has wrong size"
        self.base_image = asarray(base_image)
        self.vessels = (asarray(vessels)/255).astype(int)
        self.mask = (asarray(mask)/255).astype(int)

    def __repr__(self):
        return 'base_image: {}, vessels: {}, size: {}' \
            .format(self.base_image.filename, self.vessels.filename, self.base_image.size)
