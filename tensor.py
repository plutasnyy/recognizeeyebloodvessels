class Tensor:
    def __init__(self, base_image, vessels):
        assert base_image.size == vessels.size, "Images have different sizes"
        self.base_image = base_image
        self.vessels = vessels

    def __repr__(self):
        return 'base_image: {}, vessels: {}, size: {}' \
            .format(self.base_image.filename, self.vessels.filename, self.base_image.size)
