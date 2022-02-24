import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

img = cv.imread("../assets/sift_orig.png", cv.IMREAD_GRAYSCALE)
#cv.imshow('orig', img)

#scale the image by 60%
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

scale_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
#cv.imshow('scale', scale_img)

#rotate the image by 90 degrees
rotate_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
#cv.imshow('rotate_img', rotate_img)

#change the illumination
kernel = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])
illumination_img = cv.filter2D(img, -1, kernel)
#cv.imshow("illumination", illumination_img)

#change occlusion
img_2 = img.copy()
occlude_img = cv.rectangle(img_2, (200,200), (100, 100), 0, -1)
#cv.imshow('occlude', occlude_img)

#create sift class
sift = cv.xfeatures2d.SIFT_create()
start = time.time()
keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
end = time.time()
print(f"Orig Sift Computation time: {end - start}")

start = time.time()
keypoints_2, descriptors_2 = sift.detectAndCompute(scale_img, None)
end = time.time()
print(f"Scale Sift Computation time: {end - start}")

start = time.time()
keypoints_3, descriptors_3 = sift.detectAndCompute(rotate_img, None)
end = time.time()
print(f"Rotate Sift Computation time: {end - start}")

start = time.time()
keypoints_4, descriptors_4 = sift.detectAndCompute(illumination_img, None)
end = time.time()
print(f"Illumination Sift Computation time: {end - start}")

start = time.time()
keypoints_5, descriptors_5 = sift.detectAndCompute(occlude_img, None)
end = time.time()
print(f"Occlusion Sift Computation time: {end - start}")

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches_1 = bf.match(descriptors_1, descriptors_2)
matches_1 = sorted(matches_1,  key = lambda x:x.distance)

matches_2 = bf.match(descriptors_1, descriptors_3)
matches_2 = sorted(matches_2,  key = lambda x:x.distance)

matches_3 = bf.match(descriptors_1, descriptors_4)
matches_3 = sorted(matches_3,  key = lambda x:x.distance)

matches_4 = bf.match(descriptors_1, descriptors_5)
matches_4 = sorted(matches_4,  key = lambda x:x.distance)

scale_match = cv.drawMatches(img, keypoints_1, scale_img, keypoints_2, matches_1[:50], scale_img, flags = 2)

rotate_match = cv.drawMatches(img, keypoints_1, rotate_img, keypoints_3, matches_2[:50], rotate_img, flags = 2)

illumination_match = cv.drawMatches(img, keypoints_1, illumination_img, keypoints_4, matches_3[:50], rotate_img, flags = 2)

occlude_match = cv.drawMatches(img, keypoints_1, occlude_img, keypoints_2, matches_4[:50], occlude_img, flags = 2)

#cv.imshow('scale_match',scale_match)
#cv.imshow('rotate_match', rotate_match)
#cv.imshow('illumination_match', illumination_match)
#cv.imshow('occlude_match', occlude_match)



cv.waitKey(0)
cv.destroyAllWindows()
