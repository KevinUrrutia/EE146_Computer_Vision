import cv2 as cv

#open the image using opencv

'''
Dilate(I,H):
    Input: I, a binary image
    H, binary stucturing element
    Returns dilated image I' = I or H

    create map I' = I or H

    for all(p) in MxN:
        I'(p)<-0
    for all q in H:
        for all p in I:
            I'(p+q) <- 1
    return I'
'''

'''
Erode(I, H):
    Input: I, a binary image
    H, binary stucturing element
    Returns eroded image I' = I and H

    invert_I  = invert(I)
    H* = reflect(H)
    I' = invert(dilate(inver_I, H*))
    return I'
'''

img = cv.imread("../assets/morph_image.png", cv.IMREAD_GRAYSCALE)
cv.imshow("original", img)

#binarize the image using otsu's algorthm, see lab 2 for actual implementation of otsu algorithm
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

#get structuring element that is 5 pixels wide
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
#dialate using the stucturing element to get rid of dots in forground
dialate = cv.dilate(img, kernel, iterations=1)

#erode the image to return the rest of the image to original form
erode = cv.erode(dialate, kernel, iterations=1)

#display the new image
cv.imshow("closed image result", erode)
#note process could have been done using closing in opencv

cv.waitKey(0)
cv.destroyAllWindows()
