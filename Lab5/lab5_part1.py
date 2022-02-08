import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = cv.imread('../assets/lines.png', cv.IMREAD_GRAYSCALE)

edges = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)

lines = cv.HoughLines(edges, 1, np.pi/180, 80, None, 0,0)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img1 = img.copy()

r_arr = np.zeros([1,0])
theta_arr = np.zeros([1,0])


for i in range(0, len(lines)):
	a = np.cos(lines[i][0][1])
	b = np.sin(lines[i][0][1])

	x0 = a *lines[i][0][0]
	y0 = b*lines[i][0][0]
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))

	cv.line(img1, (x1, y1), (x2, y2), (0,0, 255), 2)
	r_arr = np.append(r_arr, lines[i][0][0])
	theta_arr = np.append(theta_arr, lines[i][0][1])

	

compare_hough = np.concatenate((img, img1), axis=1)
plt.plot(r_arr, theta_arr)
plt.show()


cv.imshow('compare hough lines', compare_hough)


cv.waitKey(0)
cv.destroyAllWindows()
