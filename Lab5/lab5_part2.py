import cv2 as cv
import numpy as np

img = cv.imread('../assets/shapesCorner.tif', cv.IMREAD_COLOR)
img1 = img.copy()
img2 = img.copy()

gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.07)
img1[dst>0.01*dst.max()] = [0,0,255]


gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 3, 5, 0.07)
img2[dst>0.01*dst.max()] = [0,0,255]

concatenated_neighbor = np.concatenate((img, img1), axis=1)
concatenated_neighbor = np.concatenate((concatenated_neighbor, img2), axis=1)

img3 = img.copy()
img4 = img.copy()

gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.07)
img3[dst>0.01*dst.max()] = [0,0,255]


gray = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 7, 0.07)
img4[dst>0.01*dst.max()] = [0,0,255]

concatenated_threshold = np.concatenate((img, img3), axis=1)
concatenated_threshold = np.concatenate((concatenated_threshold, img4), axis=1)

img5 = img.copy()
img6 = img.copy()

gray = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.07)
img5[dst>0.01*dst.max()] = [0,0,255]


gray = cv.cvtColor(img6, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.08)
img6[dst>0.01*dst.max()] = [0,0,255]

concatenated_control = np.concatenate((img, img5), axis=1)
concatenated_control = np.concatenate((concatenated_control, img6), axis=1)

cv.imshow('control parameter', concatenated_control)
cv.imshow('response threshold', concatenated_threshold)
cv.imshow('neighborhood radius', concatenated_neighbor)

cv.waitKey(0)
cv.destroyAllWindows()
