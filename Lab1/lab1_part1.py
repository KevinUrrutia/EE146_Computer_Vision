'''
Problem 1: Become familiar with image display related functions in matlab. Display a
gray scale image, color image and binary image. Find the number of rows, columns and
the number of bytes/pixel. Convert a color image to a gray scale image. What
transformation is used here?
'''

import cv2 as cv
import os

path = "../assets/"

#get image in color and display it
img_color = cv.imread("../assets/one-piece.jpg", cv.IMREAD_COLOR)
cv.imshow("one-piece color", img_color)
print("Shape of the colored image: ")
print(img_color.shape)

#Convert image to grayscale and display it
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
cv.imshow("one-piece grayscale", img_gray)
print("Shape of the gray scale image: ")
print(img_gray.shape)
cv.imwrite(os.path.join(path, "one-piece_grayscale.jpg"), img_gray)

#Convery image to binary and display it
(thresh, img_bin) = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY) 
cv.imshow("one-piece binary", img_bin)
print("Shape of the binary image: ")
print(img_bin.shape)
cv.imwrite(os.path.join(path, "one-piece_binary.jpg"), img_bin)

cv.waitKey(0)

cv.destroyAllWindows()
