import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def makeMeanTables(h):
	K = h.size
	mu0 = np.zeros([K-1, 1])
	mu1 = np.zeros([K-1, 1])
	n0 = 0
	s0 = 0

	for q in range(0, K-1):
		n0 = n0 + h[q]
		s0 = s0 + q*h[q]
		if (n0 > 0):
			mu0[q] = s0 / n0
		else:
			mu0[q] = -1

	N = n0
	
	n1 = 0
	s1 = 0
	for q in range(K-2, -1, -1):
		n1 = n1 + h[q + 1]
		s1 = s1 + (q+1)*h[q+1]
		if(n1 > 0):
			mu1[q] = s1 / n1
		else:
			mu1[q] = -1
	
	return (mu0, mu1, N)

#import and read the image, display it as well as its histogram
gray_img = cv.imread("../assets/textured2.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("grayed image", gray_img)

#use opencv algorithm to segment the image
otsu_threshold, image_result = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
print(otsu_threshold)
cv.imshow("image_result", image_result)

#calculate the histogram of the image
hist, bin_edges = np.histogram(gray_img, bins=256, density=False)
(mu0, mu1, N) = makeMeanTables(hist)

K = hist.size
var_between_max = 0
q_max = -1
n0 = 0

#store the information of var_between in an array
var_between_arr = np.zeros([K-1, 1])
#store the forground and background varience in array
var_fground_arr = np.zeros([K-1, 1])
var_bground_arr = np.zeros([K-1, 1])

for q in range(0, K-2):
	n0 = n0 + hist[q]
	n1 = N - n0
	if((n0 > 0) and (n1 > 0)):
		var_between = (n0 * n1 * ((mu0[q] - mu1[q]) ** 2)) /(N **2)
		var_between_arr[q] = var_between
		#calculate the variences of the forground
		for i in range(0, q):
			var_fground_arr[q] = (((i - mu0[q]) ** 2) * hist[i])/n0
		#calculate the varience of the background
		for i in range(q+1, K-1):
			var_bground_arr[q] = (((i - mu1[q]) ** 2) * hist[i]) /n1
		if (var_between > var_between_max):
			var_between_max = var_between
			q_max = q
print(q_max)

gray_img[gray_img>q_max] = 255
gray_img[gray_img!=255] = 0

cv.imshow("otsu algorith", gray_img)

plt.plot(hist)
plt.title("histogram")
plt.show()
plt.plot(var_fground_arr)
plt.title("foreground varience")
plt.show()
plt.plot(var_bground_arr)
plt.title("background varience")
plt.show()
plt.plot(var_between_arr)
plt.title("inter varience")
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
