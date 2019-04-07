import glob
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float, measure
from skimage import exposure
from tensor import Tensor

tensors_list = list()
images_path = sorted(glob.glob("images/*"))
for i in range(0, len(images_path), 2):
    base_image = Image.open(images_path[i])
    vessels = Image.open(images_path[i + 1])
    tensors_list.append(Tensor(base_image, vessels))


def draw_images(images: list):
    size = np.ceil(np.sqrt(len(images)))
    fig = plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        fig.add_subplot(size, size, i + 1)
        plt.imshow(img)
    plt.show()


def preprocess_image(img):
    img_eq = exposure.equalize_hist(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    draw_images(
        [img, img_eq, img_adapteq, img, img_eq, img_adapteq, img, img_eq, img_adapteq, img, img_eq, img_adapteq, img,
         img_eq, img_adapteq])


preprocess_image(tensors_list[50].base_image)
