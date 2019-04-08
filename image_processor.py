import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.filters import sobel

from tensor import Tensor


def draw_images(images: list):
    size = np.ceil(np.sqrt(len(images)))
    fig = plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        fig.add_subplot(size, size, i + 1)
        plt.imshow(img)
    plt.show()


def draw_grey_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def preprocess_image(tensor: Tensor):
    bw_image = cv2.cvtColor(tensor.base_image, cv2.COLOR_RGB2GRAY)
    image_adapt = exposure.equalize_adapthist(bw_image)
    edge_sobel = sobel(image_adapt)
    normalized_image = cv2.normalize(edge_sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    _, thresholded_image = cv2.threshold(normalized_image, 20, 255, cv2.THRESH_BINARY)
    cleaned_image = (thresholded_image * tensor.mask).astype(int)
    return np.invert(cleaned_image)
