from PIL import Image, ImageOps
from skimage.util import view_as_windows

from image_processor import draw_images, correct_image
from model import Model
import numpy as np

from utils import PATCH_SIZE, HALF_OF_PATCH_SIZE

img = Image.open('images/185.jpg')
img_with_border = correct_image(np.array(ImageOps.expand(img, border=PATCH_SIZE, fill='black')))

model = Model()
model.load_weights('weights/weights-improvement-49-0.90.hdf5')
model.compile()

patches_list = view_as_windows(img_with_border, (PATCH_SIZE, PATCH_SIZE))
predicted_img = np.zeros_like(img_with_border)

print(patches_list.shape)
for i in range(patches_list.shape[0]):
    print('i: {}'.format(i))
    for j in range(patches_list.shape[1]):
        x = i + HALF_OF_PATCH_SIZE
        y = j + HALF_OF_PATCH_SIZE
        predicted_value = np.argmax(model.predict(patches_list[i][j]))
        predicted_img[x][y] = predicted_value

img_without_border = ImageOps.crop(Image.fromarray(predicted_img), PATCH_SIZE)
draw_images([img_without_border])
