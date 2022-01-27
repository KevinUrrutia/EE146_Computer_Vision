import cv2 as cv
import numpy as np

img = cv.imread('../assets/cameramanBlur.tif', cv.IMREAD_GRAYSCALE)


#canny
canny1 = cv.Canny(img, 50, 100)
canny2 = cv.Canny(img, 100, 200)
canny3 = cv.Canny(img, 200, 300)
compare_canny = np.concatenate((img, canny1), axis=1)
compare_canny = np.concatenate((compare_canny, canny2), axis=1)
compare_canny = np.concatenate((compare_canny, canny3), axis=1)
cv.imshow('canny comaprison', compare_canny)


#sobel
sobel1 = cv.Sobel(img, cv.CV_8U, 1, 1, ksize=5)
sobel2 = cv.Sobel(img, cv.CV_8U, 2, 2, ksize=5)
sobel3 = cv.Sobel(img, cv.CV_8U, 3, 3, ksize=5)
compare_sobel = np.concatenate((img, sobel1), axis=1)
compare_sobel = np.concatenate((compare_sobel, sobel2), axis=1)
compare_sobel = np.concatenate((compare_sobel, sobel3), axis=1)
cv.imshow('sobel comparison', compare_sobel)

#prewit
kernel1 = np.array([
  [-1, -1, -1],
  [-1, 8, -1],
  [-1, -1, -1]
])
kernel2 = np.array([
  [-2, -2, -2],
  [-2, 16, -2],
  [-2, -2, -2]
])
kernel3 = np.array([
  [-3, -3, -3],
  [-3, 24, -3],
  [-3, -3, -3]
])
prewit1 = cv.filter2D(img, -1, kernel1)
prewit2 = cv.filter2D(img, -1, kernel2)
prewit3 = cv.filter2D(img, -1, kernel3)
compare_prewit = np.concatenate((img, prewit1), axis=1)
compare_prewit = np.concatenate((compare_prewit, prewit2), axis=1)
compare_prewit = np.concatenate((compare_prewit, prewit3), axis=1)
cv.imshow('prewit comparison', compare_prewit)


#LOG
kernel = np.array([
  [0, -1, 0],
  [-1, 4, -1],
  [0, -1, 0]
])
laplacian1 = cv.Laplacian(img, cv.CV_8U, ksize=3)
laplacian2 = cv.filter2D(img, -1, kernel)
compare_laplacian = np.concatenate((img, laplacian1), axis=1)
compare_laplacian = np.concatenate((compare_laplacian, laplacian2), axis=1)
cv.imshow('laplacian comparison', compare_laplacian)

cv.waitKey(0)
cv.destroyAllWindows()
