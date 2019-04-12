import glob
from PIL import Image

from tensor import Tensor


def create_tensor_from_file(one_tensor=False):
    tensors_list = list()
    images_path = sorted(glob.glob("images/*"))
    for i in range(0, len(images_path), 2):
        base_image = Image.open(images_path[i])
        vessels = Image.open(images_path[i + 1])
        file_name = images_path[i].replace('/', '.').split('.')[1]
        mask = Image.open('masks/{}.jpg'.format(file_name))
        tensors_list.append(Tensor(base_image, vessels, mask))
        if one_tensor:
            break
    return tensors_list
