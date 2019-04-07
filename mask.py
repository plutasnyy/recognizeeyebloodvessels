import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

for i in range(131,187,2):
    im = Image.open('images/'+str(i)+'.jpg')
    width, height = im.size


    img = cv2.imread('images/'+str(i)+'.jpg',0)
    ret,thresh1 = cv2.threshold(img,10,255,cv2.THRESH_BINARY)


    image = im = Image.fromarray(np.uint8(thresh1))
    new_image = image.resize((width,height))
    new_image = new_image.convert("RGB")
    new_image.save(str(i)+'mask.jpg')