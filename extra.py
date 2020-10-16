import cv2 as cv

img = cv.imread(cv.samples.findFile("football.jpeg"))

if img is None:
    sys.exit("Could not read the image.")
    
cv.imshow("Display window", img)

red_guy = img[50:300, 250:350]
blue_guy = img[50:300, 70:200]
ball = img[190:250, 200:250]
cv.imshow("Red_guy", red_guy)
cv.imshow("Blue_guy", blue_guy)
cv.imshow("Ball", ball)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("football.jpeg", img)