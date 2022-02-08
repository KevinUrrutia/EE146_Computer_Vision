import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = cv.imread('../assets/lines.png', cv.IMREAD_GRAYSCALE)

edges = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)

lines1 = cv.HoughLinesP(edges, 1, np.pi/180, 10, None, 0, 0)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img1 = img.copy()

r = np.zeros([1,0])
theta = np.zeros([1,0])
for line in lines1:
    x1, y1, x2, y2 = line[0]
    cv.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    r = np.append(r, x1)
    theta = np.append(theta, x2)

compare_hough = np.concatenate((img, img1), axis=1)
#plt.plot(theta, r)
#plt.show()

cv.imshow('compare hough lines', compare_hough)


cv.waitKey(0)
cv.destroyAllWindows()
