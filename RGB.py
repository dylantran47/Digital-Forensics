import cv2 as cv
import glob
import os

img = cv.imread(cv.samples.findFile("logo-usth.png"))


#if img is None:
 #   sys.exit("Could not read the image.")

cv.imshow("Display window", img)

b, g, r = cv.split(img)
cv.imshow('imge1', r)
#cv.waitKey(0)
cv.imshow('imge2', g)
#cv.waitKey(0)
cv.imshow('imge3', b)
#cv.waitKey(0)

#cv.destroyAllWindows()

image = cv.merge((r,g,b))

cv.imshow('image', image)
cv.waitKey(0)

cv.destroyAllWindows()