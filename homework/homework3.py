'''
calculate the distance between each pair of points
then take the max of these distances


def maximalDiameter(points):
    
    Inputs: set of points in the figure listed as an array
    Returns the maximum distance between the pairs of points
    

    #set up an array that will hold distances between pairs of points
    distances = np.zeros[[1,1]]

    for i in range(points.shape):
        for j in range(i+1, points.shape-1):
            distances = np.append(distances, abs(points[i] - points[j]))

    max = np.max(distances)

    return max
'''
'''
import cv2 as cv

img = cv.imread('../assets/hu_moments_orig.png', cv.IMREAD_GRAYSCALE)

otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

moments1 = cv.moments(bin_img)
hu_moments1 = cv.HuMoments(moments1)

rot_img = cv.rotate(bin_img, cv.ROTATE_90_CLOCKWISE)

moments2 = cv.moments(rot_img)
hu_moments2 = cv.HuMoments(moments2)

print(hu_moments1)

print(hu_moments2)

cv.waitKey(0)
cv.destroyAllWindows()
'''

import cv2 as cv
import numpy as np

img = cv.imread('../assets/gears.png', cv.IMREAD_GRAYSCALE)
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bin_img, 4, cv.CV_32S)
for i in range(1, num_labels):
    (cX, cY) = centroids[i]
    cv.circle(bin_img, (int(cX), int(cY)), 155, (0,0,255), -1)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
dilate = cv.dilate(bin_img, kernel, iterations =8)
cv.imshow('dilate', dilate)

erode = cv.erode(dilate, kernel, iterations=4)
cv.imshow('close', erode)


cv.imshow('orig', bin_img)

cv.waitKey(0)
cv.destroyAllWindows()

