import logging

from PIL import Image, ImageOps
from PIL.Image import NEAREST
from skimage.util import view_as_windows

from image_processor import draw_images, correct_image
from model import Model
import numpy as np

from utils import PATCH_SIZE, HALF_OF_PATCH_SIZE, LONG_EDGE_SIZE

logging.basicConfig(level=logging.INFO)
img = Image.open('test/009.jpg')
long_edge = max(np.array(img).shape)
scale = LONG_EDGE_SIZE / long_edge
w, h, c = np.array(img).shape
w, h = int(w * scale), int(h * scale)
img = img.resize((w, h, c), resample=NEAREST)
img_with_border = correct_image(np.array(ImageOps.expand(img, border=PATCH_SIZE, fill='black')))

model = Model()
model.load_weights('best.hdf5')
model.compile()

patches_list = view_as_windows(img_with_border, (PATCH_SIZE, PATCH_SIZE))
predicted_img = np.zeros_like(img_with_border)
logging.info('Created patches')

for i in range(patches_list.shape[0]):
    logging.info('i: {}'.format(i))
    for j in range(patches_list.shape[1]):
        x = i + HALF_OF_PATCH_SIZE
        y = j + HALF_OF_PATCH_SIZE
        predicted_value = np.argmax(model.predict(patches_list[i][j]))
        predicted_img[x][y] = predicted_value

# TODO crop doesnt work yet
img_without_border = ImageOps.crop(Image.fromarray(predicted_img), PATCH_SIZE)
draw_images([predicted_img])
