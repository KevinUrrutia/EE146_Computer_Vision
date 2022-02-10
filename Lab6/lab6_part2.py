import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = cv.imread('../assets/peppers_trees.png', cv.IMREAD_UNCHANGED)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

pixel_val = img.reshape((-1,3))
print(pixel_val.shape)
pixel_val = np.float32(pixel_val)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
retval, labels, centers = cv.kmeans(pixel_val, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()

segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((img.shape))

plt.imshow(segmented_image)
plt.show()
