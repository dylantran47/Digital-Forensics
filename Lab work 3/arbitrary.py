import cv2 as cv
from matplotlib import pyplot as plt


forme1 = cv.imread("forme1.png", cv.IMREAD_GRAYSCALE)
house8 = cv.imread("house8.png", cv.IMREAD_GRAYSCALE)
woman = cv.imread("femme.png", cv.IMREAD_GRAYSCALE)

ret, forme1_thresh_bin = cv.threshold(forme1, 128, 255, cv.THRESH_BINARY)
ret, forme1_thresh_bin_inv = cv.threshold(forme1, 128, 255, cv.THRESH_BINARY_INV)
ret, house8_thresh_bin = cv.threshold(house8, 96, 255, cv.THRESH_BINARY)
ret, house8_thresh_bin_inv = cv.threshold(house8, 96, 255, cv.THRESH_BINARY_INV)
ret, woman_thresh_bin = cv.threshold(woman, 95, 255, cv.THRESH_BINARY)
ret, woman_thresh_bin_inv = cv.threshold(woman, 95, 255, cv.THRESH_BINARY_INV)


titles1 = ['f1', 'f2', 'f3']
images1 = [forme1, forme1_thresh_bin, forme1_thresh_bin_inv]
for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images1[i], 'gray')
    plt.title(titles1[i])
    plt.xticks([]), plt.yticks([])
plt.show()

titles2 = ['h1', 'h2', 'h3']
images2 = [house8, house8_thresh_bin, house8_thresh_bin_inv]
for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images2[i], 'gray')
    plt.title(titles2[i])
    plt.xticks([]), plt.yticks([])
plt.show()

titles3 = ['w1', 'w2', 'w3']
images3 = [woman, woman_thresh_bin, woman_thresh_bin_inv]
for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images3[i], 'gray')
    plt.title(titles3[i])
    plt.xticks([]), plt.yticks([])
plt.show()
