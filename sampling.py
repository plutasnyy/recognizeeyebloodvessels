import numpy as np
from PIL import Image
import glob

array = []
w = 1
images_path = sorted(glob.glob("images/*"))
images_path2 = sorted(glob.glob("masks/*"))
for i in range(1,186,2):
    array2 = []
    im = np.asarray(Image.open(images_path2[w]))
    w+=1
    image = np.asarray(Image.open(images_path[i]))
    for x in range(len(im[0])-30):
        for y in range(len(im)-30):
            if image[y+16][x+16].all() == 0:
                continue
            else:
                print([y,y+31],[x,x+31])
                array2.append([[y,y+31],[x,x+31]])
    array.append(array2)
