import cv2 as cv
import numpy as np
from skimage import measure
import math

img = np.array([[0, 0, 1, 0, 0, 1, 1, 1], 
		[0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0],
		[1, 1, 1, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 0, 1, 1, 1]])

labels = measure.label(img, neighbors=8, background=0)
print(labels)

img = np.array([[1, 1, 0, 1, 1, 1, 0, 1], 
		[1, 1, 0, 1, 0, 1, 0, 1],
		[1, 1, 1, 1, 0, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 1],
		[1, 1, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 1],
		[1, 1, 0, 1, 0, 0, 0, 1],
		[1, 1, 0, 1, 0, 1, 1, 1]])

labels = measure.label(img, neighbors=8, background=0)
print(labels)

img = cv.imread("../assets/shapes.png", cv.IMREAD_GRAYSCALE)
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bin_img , 4 , cv.CV_32S)
label_colors = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_colors)

labeled_img = cv.merge([label_colors, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
labeled_img[label_colors==0] = 0

area = np.zeros([num_labels, 1])
perimeter = np.zeros([num_labels, 1])
circularity = np.zeros([num_labels, 1])

for i in range(1, num_labels):
	area[i] = stats[i, cv.CC_STAT_AREA]
	perimeter[i] = (2 * stats[i, cv.CC_STAT_WIDTH]) + (2 * stats[i, cv.CC_STAT_HEIGHT])
	circularity[i] = (4 * area[i]) / (perimeter[i] ** 2)
	(cX, cY) = centroids[i]
	cv.circle(labeled_img, (int(cX), int(cY)), 4, (0,0,255), -1)

print("Area")
print(area)
print("Perimeter")
print(perimeter)
print("Circularity")
print(circularity)
cv.imshow("labeled images", labeled_img)

cv.waitKey(0)
cv.destroyAllWindows()
