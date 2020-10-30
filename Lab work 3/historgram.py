import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("forme3.png", cv.IMREAD_GRAYSCALE)


cv.imshow("image", img)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()