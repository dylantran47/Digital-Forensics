import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(cv.samples.findFile("cadastre2.png"), cv.IMREAD_GRAYSCALE)

ret, mask = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


kernel = np.ones((11,11), np.uint8)
kernel2 = np.ones((1,1), np.uint8)
dila = cv.dilate(mask, kernel2, iterations=2)
clos = cv.morphologyEx(dila, cv.MORPH_CLOSE, kernel)

titles = ['image', 'mask', 'dilation', 'close']
images = [img, mask, dila, clos]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()