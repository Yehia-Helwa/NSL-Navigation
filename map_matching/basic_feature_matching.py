import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
#approx 5.2 second running time
start = time.time()
MIN_MATCH_COUNT = 10

def resize(scale,img):
    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

img1 = cv.imread('data/sweden/sweden_full.png',1)          # queryImage
img1=resize(10,img1)
img2 = cv.imread('data/sweden/sweden_test.JPG',1) # trainImage
img2=resize(10,img2)


plt.imshow(img1)
plt.figure()
plt.imshow(img2)
plt.show()
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher()

print(len(des1))
print(len(des2))

matches = bf.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

# Apply ratio test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.5*n.distance:
        matchesMask[i]=[1,0]

# cv.drawMatchesKnn expects list of lists as matches.


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 2)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
img4 = cv.drawKeypoints(img1, kp1, None, color=(255,0,0))
img5 = cv.drawKeypoints(img2, kp2, None, color=(255,0,0))
plt.figure()
plt.imshow(img3)
# plt.figure()
# plt.imshow(img4)
# plt.figure()
# plt.imshow(img5)
plt.show()
