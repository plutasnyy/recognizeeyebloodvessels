import glob
from PIL import Image

from tensor import Tensor

tensors_list = list()
images_path = sorted(glob.glob("images/*"))
for i in range(0, len(images_path), 2):
    base_image = Image.open(images_path[i])
    vessels = Image.open(images_path[i + 1])
    tensors_list.append(Tensor(base_image, vessels))

for tensor in tensors_list:
    print(tensor)
