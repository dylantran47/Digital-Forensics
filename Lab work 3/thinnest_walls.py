import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load image
img = cv.imread(cv.samples.findFile("cadastre2.png"), cv.IMREAD_GRAYSCALE)
# make binary image
ret, mask = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

##### make thickest walls 
kernel = np.ones((11,11), np.uint8)
kernel2 = np.ones((1,1), np.uint8)
dila = cv.dilate(mask, kernel2, iterations=2)
clos = cv.morphologyEx(dila, cv.MORPH_CLOSE, kernel)
clos_inv = np.invert(clos)

## make thinnest walls
# delete delete stairs
struc1 = cv.getStructuringElement(cv.MORPH_RECT, (5,6))
clos2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, struc1)

# use insert image in opencv to delete the same details (thickest walls)
img2_fg = cv.bitwise_and(clos_inv, clos_inv) #Take only region object of 
imgadd = cv.add(clos2, img2_fg) #put the object in the image has been deleted stairs
imgadd_inv = np.invert(imgadd)
thinnest_wall_closing = cv.morphologyEx(imgadd_inv, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
#####

titles = ['image', 'mask', 'thickest_wall_closing_inv', 'thinnest_wall_closing']
images = [img, mask, clos_inv, thinnest_wall_closing]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()