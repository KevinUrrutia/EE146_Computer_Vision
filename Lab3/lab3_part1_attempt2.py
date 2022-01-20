import cv2 as cv
import numpy as np


bin_img = cv.imread("../assets/circles.png", cv.IMREAD_UNCHANGED)

kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

erosion1 = cv.erode(bin_img, kernel1, iterations=1)
dilate1 = cv.dilate(bin_img, kernel1, iterations=1)
opening1 = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel1)
closing1 = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel1)
img1 = np.concatenate((erosion1, dilate1, opening1, closing1), axis=1)

kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
erosion2 = cv.erode(bin_img, kernel2, iterations=1)
dilate2 = cv.dilate(bin_img, kernel2, iterations=1)
opening2 = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel2)
closing2 = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel2)
img2 = np.concatenate((erosion2, dilate2, opening2, closing2), axis=1)

kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
erosion3 = cv.erode(bin_img, kernel3, iterations=1)
dilate3 = cv.dilate(bin_img, kernel3, iterations=1)
opening3 = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel3)
closing3 = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel3)
img3 = np.concatenate((erosion3, dilate3, opening3, closing3), axis=1)

kernel4 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
erosion4 = cv.erode(bin_img, kernel4, iterations=1)
dilate4 = cv.dilate(bin_img, kernel4, iterations=1)
opening4 = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel4)
closing4 = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel4)
img4 = np.concatenate((erosion4, dilate4, opening4, closing4), axis=1)

img = np.concatenate((img1, img2, img3, img4), axis=0)
cv.imwrite('../assets/Lab3_part1.png', img)

cv.imshow("Morph operations", img)

cv.waitKey(0)
cv.destroyAllWindows()
