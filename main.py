import glob
from PIL import Image

from image_processor import preprocess_image, draw_grey_image
from tensor import Tensor

tensors_list = list()
images_path = sorted(glob.glob("images/*"))
for i in range(0, len(images_path), 2):
    base_image = Image.open(images_path[i])
    vessels = Image.open(images_path[i + 1])
    file_name = images_path[i].replace('/', '.').split('.')[1]
    mask = Image.open('masks/{}.jpg'.format(file_name))
    tensors_list.append(Tensor(base_image, vessels, mask))

processed_image = preprocess_image(tensors_list[2])
draw_grey_image(processed_image)
