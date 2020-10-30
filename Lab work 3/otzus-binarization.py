import cv2 as cv
import matplotlib.pyplot as plt

forme1 = cv.imread("forme1.png", cv.IMREAD_GRAYSCALE)
forme3 = cv.imread("forme3.png", cv.IMREAD_GRAYSCALE)
lena = cv.imread("lena.png", cv.IMREAD_GRAYSCALE)

titles1 = ['forme1', 'forme3', 'lena']
images1 = [forme1, forme3, lena]

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images1[i], 'gray')
    plt.title(titles1[i])
    plt.xticks([]), plt.yticks([])
plt.show()

ret, otzu1 = cv.threshold(forme1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
ret, otzu3 = cv.threshold(forme3, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
ret, otzu_lena = cv.threshold(lena, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

titles2 = ['otzu1', 'otzu3', 'otzu_lena']
images2 = [otzu1, otzu3, otzu_lena] 

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images2[i], 'gray')
    plt.title(titles2[i])
    plt.xticks([]), plt.yticks([])
plt.show()

blur_forme1 = cv.GaussianBlur(forme1, (5, 5), 0)
ret, gaussian_otzu1 = cv.threshold(blur_forme1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

blur_forme3 = cv.GaussianBlur(forme3, (5, 5), 0)
ret, gaussian_otzu3 = cv.threshold(blur_forme3,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

blur_lena = cv.GaussianBlur(lena, (5, 5), 0)
ret, gaussian_otzu_lena = cv.threshold(blur_lena,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

titles3 = ['gaussian_otzu1', 'gaussian_otzu3', 'gaussian_otzu_lena']
images3 = [gaussian_otzu1, gaussian_otzu3, gaussian_otzu_lena]

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images3[i], 'gray')
    plt.title(titles3[i])
    plt.xticks([]), plt.yticks([])
plt.show()
