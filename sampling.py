import numpy as np
from PIL import Image

array2 = []
i=1
im = Image.open('images/00'+str(i)+'.jpg')

width, height = im.size
im = np.array(im)
image = Image.open('mask/00'+str(i)+'mask.jpg')

image = np.array(image)

for q in range(width-30):
    for y in range(height-30):
        print(q)
        if image[y+16][q+16].all() == 0:
            continue
        else:
            print([y,y+31],[q,q+31])
            array2.append([[y,y+31],[q,q+31]])

