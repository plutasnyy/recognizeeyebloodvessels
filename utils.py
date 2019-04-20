import glob
import logging

from PIL import Image

from tensor import Tensor

PATCH_SIZE = 32
HALF_OF_PATCH_SIZE = int(PATCH_SIZE / 2)


def create_tensor_from_file(one_tensor=False):
    logging.info('Create tensor list, only one tensor is {}'.format(one_tensor))
    tensors_list = list()
    images_path = sorted(glob.glob("images/*"))
    for i in range(0, len(images_path), 2):
        # TODO FIX for i=164
        base_image = Image.open(images_path[i])
        vessels = Image.open(images_path[i + 1])
        file_name = images_path[i].replace('/', '.').split('.')[1]
        mask = Image.open('masks/{}.jpg'.format(file_name))
        tensors_list.append(Tensor(base_image, vessels, mask, i))
        tensors_list[0].draw_tensor()
        if one_tensor:
            break
    return tensors_list
