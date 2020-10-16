import cv2 as cv
import numpy as np

img = cv.imread(cv.samples.findFile("logo-usth.png"))

rgb = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    

# red shape
lower_red = np.array([161, 155, 84])
upper_red = np.array([179, 255, 255])
red_mask = cv.inRange(rgb, lower_red, upper_red)
red = cv.bitwise_and(img, img, mask=red_mask)

# blue shape
lower_blue = np.array([94, 80, 2])
upper_blue = np.array([126, 255, 255])
blue_mask = cv.inRange(rgb, lower_blue, upper_blue)
blue = cv.bitwise_and(img, img, mask=blue_mask)



cv.imshow("Display window", img)
cv.imshow("Red", red)
cv.imshow("Blue", blue)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("football.jpeg", img)