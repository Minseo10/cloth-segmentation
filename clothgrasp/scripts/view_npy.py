import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('/home/minseo/robot_ws/src/cloth-segmentation/clothgrasp/results/edges/network/03_04_2024_17:08:18:214627/1_direction_x.npy')


plt.imshow(img_array, cmap='gray')
plt.show()