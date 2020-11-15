import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Load file
img = cv.imread('LaRochelle.jpg')

#Convert image to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Convert image to gray scale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

fast = cv.FastFeatureDetector_create() 

#Detect keypoints with non max suppression
keypoints_with_nonmax = fast.detect(gray, None)

#Disable nonmaxSuppression 
fast.setNonmaxSuppression(False)

#Detect keypoints without non max suppression
keypoints_without_nonmax = fast.detect(gray, None)

image_with_nonmax = np.copy(img)
image_without_nonmax = np.copy(img)

#Draw keypoints on top of the input image
cv.drawKeypoints(img, keypoints_with_nonmax, image_with_nonmax, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.drawKeypoints(img, keypoints_without_nonmax, image_without_nonmax, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Image With Non Max Suppression: ", len(keypoints_with_nonmax))

#Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Image Without Non Max Suppression: ", len(keypoints_without_nonmax))

#Display images
titles = ['Original', 'Gray', 'With non max suppression', 'Without non max suppression' ]
images = [img, gray, image_with_nonmax, image_without_nonmax]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()
