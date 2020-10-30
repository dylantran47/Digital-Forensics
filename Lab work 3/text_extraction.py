import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load two images
img = cv.imread('cadastre1.png',0)
img = cv.medianBlur(img,5)

# make binary image
mask = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY_INV,21,14)

# make kernel through the image
kernel = np.ones((5,5), np.uint8)
kernel2 = np.ones((10,10), np.uint8)
dilation = cv.dilate(mask,kernel,iterations = 2)
# use opening morphology
opening = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel2)

titles = ['Image',
        'mask', 
        'Extra']
images = [img, mask, opening]

for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()