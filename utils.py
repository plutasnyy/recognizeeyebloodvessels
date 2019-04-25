import glob
import logging

from PIL import Image
from tensor import Tensor

PATCH_SIZE = 32
HALF_OF_PATCH_SIZE = int(PATCH_SIZE / 2)


def create_tensor_from_file(one_tensor=False):
    logging.info('Create tensor list, only one tensor is {}'.format(one_tensor))
    tensors_list = list()
    images_path = sorted(glob.glob("train/*"))
    for i in range(0, len(images_path), 3):
        file_name = images_path[i].replace('/', '.').split('.')[1]
        logging.info("Process filepath: {} with {} filename for i {}".format(images_path[i], file_name, i))
        base_image = Image.open(images_path[i])
        print(images_path[i + 2])
        mask = Image.open(images_path[i + 1])
        vessels = Image.open(images_path[i + 2])
        tensors_list.append(Tensor(base_image, vessels, mask, i))
        tensors_list[-1].draw_tensor()
        if one_tensor:
            break
    return tensors_list
