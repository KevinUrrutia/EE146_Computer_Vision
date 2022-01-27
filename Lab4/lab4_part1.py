import cv2 as cv
import numpy as np


img = cv.imread('../assets/cameramanSPN.tif', cv.IMREAD_GRAYSCALE)

median = cv.medianBlur(img, 5)
compare_median = np.concatenate((img, median), axis=1)

mean = cv.blur(img, (5,5))
compare_mean = np.concatenate((img, mean), axis=1)

gaussian = cv.GaussianBlur(img, (5,5), 0)
compare_gaussian = np.concatenate((img, gaussian), axis=1)

full = np.concatenate((compare_mean, compare_median), axis=0)
full = np.concatenate((full, compare_gaussian), axis=0)

cv.imshow("comparisons", full)

cv.waitKey(0)
cv.destroyAllWindows
