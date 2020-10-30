import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Load image
img = cv.imread(cv.samples.findFile("cadastre2.png"), cv.IMREAD_GRAYSCALE)

#create binary image
ret, mask = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# make kernel through the image
kernel = np.ones((11,11), np.uint8)
kernel2 = np.ones((1,1), np.uint8)
# use dilation morphology
thickest_wall_dilation = cv.dilate(mask, kernel2, iterations=2)
# use closing morphology
thickest_wall_closing = cv.morphologyEx(thickest_wall_dilation, cv.MORPH_CLOSE, kernel)
thickest_wall_closing_inv = np.invert(thickest_wall_closing)


titles = ['image', 'mask', 'dilation', 'closing']
images = [img, mask, thickest_wall_dilation, thickest_wall_closing_inv]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()