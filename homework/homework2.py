import cv2 as cv

#open the image using opencv
img = cv.imread("../assets/morph_image.png", cv.IMREAD_GRAYSCALE)
cv.imshow("original", img)

#binarize the image using otsu's algorthm, see lab 2 for actual implementation of otsu algorithm
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

#get structuring element that is 5 pixels wide
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
#dialate using the stucturing element to get rid of dots in forground
dialate = cv.dilate(img, kernel, iterations=1)
cv.imshow("dilate", dialate)

#erode the image to return the rest of the image to original form
erode = cv.erode(dialate, kernel, iterations=1)

#display the new image
cv.imshow("closed image result", erode)
#note process could have been done using closing in opencv

cv.waitKey(0)
cv.destroyAllWindows()
