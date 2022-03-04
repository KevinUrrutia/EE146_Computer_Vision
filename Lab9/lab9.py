import cv2 as cv
import numpy as np
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt

img = cv.imread("../assets/rice.png", cv.IMREAD_GRAYSCALE)
#cv.imshow("orig", img)

#img = np.array(img.convert('L', colors=8))

#get the three co-occurance matrices and normalize them
gcom1 = skimage.feature.greycomatrix(img, [1,2],angles=[0, 0, 0], normed=True)
gcom2 = skimage.feature.greycomatrix(img, [2,2],angles=[0, 0, 0], normed=True)
gcom3 = skimage.feature.greycomatrix(img, [2,3],angles=[0, 0, 0], normed=True)

result1 = gcom1[:, :, 0, 0]
result2 = gcom2[:, :, 0, 0]
result3 = gcom3[:, :, 0, 0]

#display normalized co-occurance matrices
plt.imshow(result1)
plt.show()
plt.imshow(result2)
plt.show()
plt.imshow(result3)
plt.show()


#display features
contrast1 = skimage.feature.greycoprops(gcom1, 'contrast')
energy1 = skimage.feature.greycoprops(gcom1, 'energy')
correlation1 = skimage.feature.greycoprops(gcom1, 'correlation')
entropy1 = skimage.measure.shannon_entropy(result1)
print(entropy1)

contrast2 = skimage.feature.greycoprops(gcom3, 'contrast')
energy2 = skimage.feature.greycoprops(gcom3, 'energy')
correlation2 = skimage.feature.greycoprops(gcom3, 'correlation')
entropy2 = skimage.measure.shannon_entropy(result2)
print(entropy2)

contrast3 = skimage.feature.greycoprops(gcom3, 'contrast')
energy3 = skimage.feature.greycoprops(gcom3, 'energy')
correlation3 = skimage.feature.greycoprops(gcom3, 'correlation')
entropy3 = skimage.measure.shannon_entropy(result3)
print(entropy3)


cv.waitKey(0)
cv.destroyAllWindows()
