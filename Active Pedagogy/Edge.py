import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

scale = 1
delta = 0
ddepth = cv.CV_16S

#Load file
img = cv.imread('lena.png', cv.IMREAD_COLOR)

img = cv.GaussianBlur(img, (3, 3), 0)

#Convert image to RGB
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Grandient X
grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(grad_x)

#Gradient Y
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
abs_grad_y = cv.convertScaleAbs(grad_y)

sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#Canny edge detection
canny = cv.Canny(gray,50,100)

#Display images
titles = ['Original', 'Sobel', 'Canny' ]
images = [gray, sobel, canny]

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

