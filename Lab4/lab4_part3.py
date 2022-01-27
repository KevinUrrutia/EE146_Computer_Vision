import cv2 as cv
import numpy as np

img = cv.imread('../assets/cameramanBlur.tif', cv.IMREAD_GRAYSCALE)

kernel1 = np.array([
  [-1, -1, -1],
  [-1, 9, -1],
  [-1, -1, -1]
])

sharpen1 = cv.filter2D(img, -1, kernel1)

kernel2 = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])

sharpen2 = cv.filter2D(img, -1, kernel2)


sharpen_compare = np.concatenate((img, sharpen1), axis=1)
sharpen_compare = np.concatenate((sharpen_compare, sharpen2), axis=1)

cv.imshow('sharpen compare', sharpen_compare)

cv.waitKey(0)
cv.destroyAllWindows()
