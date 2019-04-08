import glob
import os

import cv2
import numpy as np
from PIL import Image

images_path = sorted(glob.glob("images/*"))
for i in range(0, len(images_path), 2):
    path = images_path[i]
    path_without_extension = os.path.splitext(os.path.basename(path))[0]
    image = np.asarray(Image.open(path))
    blured_image = cv2.GaussianBlur(image, (15, 15), 0)
    _, thresholded_image = cv2.threshold(cv2.cvtColor(blured_image, cv2.COLOR_RGB2GRAY), 28, 255, cv2.THRESH_BINARY)
    cv2.imwrite('masks/{}.jpg'.format(path_without_extension), thresholded_image)
