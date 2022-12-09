import cv2
import numpy as np


path = '/home/baihy/my_code/depth-NeRF/1_Depth_329.raw'

img = np.fromfile(path, dtype='uint8')
img = img.reshape(360, -1, 2)[:, :, 0]
# img = cv2.imread(path)
print(img.max())
print(img.min())
print(img.shape)

cv2.imwrite('depth.png', img)