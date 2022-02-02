import cv2 as cv
import numpy as np


img = cv.imread('../assets/shapesCorner.tif', cv.IMREAD_GRAYSCALE)

edges = cv.Canny(img, 50, 100)

lines1 = cv.HoughLinesP(edges, 1, np.pi/180, 10, None, 0, 0)
lines2 = cv.HoughLinesP(edges, 1, np.pi/180, 50, None, 0, 0)
lines3 = cv.HoughLinesP(edges, 1, np.pi/180, 100, None, 0, 0)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()

for line in lines1:
    x1, y1, x2, y2 = line[0]
    cv.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)

for line in lines2:
    x1, y1, x2, y2 = line[0]
    cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
for line in lines3:
    x1, y1, x2, y2 = line[0]
    cv.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 2)

compare_hough = np.concatenate((img, img1), axis=1)
compare_hough = np.concatenate((compare_hough, img2), axis=1)
compare_hough = np.concatenate((compare_hough, img3), axis=1)

cv.imshow('compare hough lines', compare_hough)
cv.waitKey(0)
cv.destroyAllWindows()
