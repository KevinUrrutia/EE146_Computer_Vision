'''
Problem 2: Histograms of images are commonly used in computer vision algorithms. In
this problem, you select a couple of images of each kind - gray scale, binary and color.
(a) Compute and display histogram of a high contrast gray scale image and a low
contrast gray scale image. What do you observe?
(b) Compute and display histogram of a binary image. What are your observations?
(c) (i) Compute and display the histogram of a color image by its individual color
channels. (ii) Now display the combined histogram of a 24 bit color image by
concatenating the values of R, G, B channels/images in a single histogram vector
of length 256x3 rather than having three vectors, each of length 256. This can be
useful in image matching. (iii) How will you obtain the 8 bit histogram of a color
image? Display your histogram. Hint: Select 3 most significant bits from Red and
Green and two most significant bits from Blue channel of a color image. Form a
new 8 bit representation of a color image by obtaining new pixel values for all the
pixels in an image. You can display this image as a gray scale 8 bit image. It is an
approximation of a 24 bit color image that is useful for a comparison of two color
images for efficient matching.
(d) Compute and display cumulative histogram of an image. Hint: Use normalized
histogram to obtain cumulative histogram.
(e) Compute statistical properties of an image: Mean, Variance and Median. 
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from dec2bin import *

#Compute and display histogram of high contrast image
high_contrast_img = cv.imread("../assets/high_contrast.jpeg", cv.IMREAD_GRAYSCALE)
plt.hist(high_contrast_img.ravel(), 256, [0,256])
plt.title("High Contrast Image")
plt.show()
#cv.imshow("high constrast image", high_contrast_img)

#Compute and display histogram of low contrast image
low_contrast_img = cv.imread("../assets/low_contrast.jpeg", cv.IMREAD_GRAYSCALE)
plt.hist(low_contrast_img.ravel(), 256, [0,256])
plt.title("Low Contrast Image")
plt.show()
#cv.imshow("Low constrast image", low_contrast_img)

#Compute and display a histogram of a binary image
(thresh, img_bin) = cv.threshold(high_contrast_img, 127, 255, cv.THRESH_BINARY) 
plt.hist(img_bin.ravel(), 256, [0,256])
plt.title("Binary_image")
plt.show()
#cv.imshow("binary image", img_bin)

#Compute and display the histogram of a color image by its individual color channels.
color_img = cv.imread("../assets/one-piece.jpg", cv.IMREAD_COLOR)
blue_channel = color_img[:,:,0]
green_channel = color_img[:,:,1]
red_channel = color_img[:,:,2]

plt.hist(blue_channel.ravel(), 256, [0,256])
plt.title("Blue Channel")
plt.show()
#cv.imshow("blue channel", blue_channel)

plt.hist(green_channel.ravel(), 256, [0,256])
plt.title("green Channel")
#plt.show()
#cv.imshow("green channel", green_channel)

plt.hist(red_channel.ravel(), 256, [0,256])
plt.title("red Channel")
#plt.show()
#cv.imshow("red channel", red_channel)
#cv.imshow("color image", color_img)

'''
How will you obtain the 8 bit histogram of a color
image? Display your histogram. Hint: Select 3 most significant bits from Red and
Green and two most significant bits from Blue channel of a color image. Form a
new 8 bit representation of a color image by obtaining new pixel values for all the
pixels in an image. You can display this image as a gray scale 8 bit image. It is an
approximation of a 24 bit color image that is useful for a comparison of two color
images for efficient matching.
'''
#Convert r, g, b matrix to binary
rows = red_channel.shape[0]
cols = red_channel.shape[1] 

red_channel = red_channel.astype(int)
green_channel = green_channel.astype(int)
blue_channel = green_channel.astype(int)

bin_red = np.zeros((rows, cols), dtype='S8')
bin_green = np.zeros((rows, cols), dtype='S8')
bin_blue = np.zeros((rows, cols), dtype='S8')
for i in range(0, rows):
	for j in range(0, cols):
		bin_red[i][j] = dec2bin(red_channel[i][j])
		bin_green[i][j] = dec2bin(green_channel[i][j])
		bin_blue[i][j] = dec2bin(blue_channel[i][j])

bin_color = np.zeros((rows, cols), dtype='S8')
for i in range(0, rows):
	for j in range(0, cols):
		bin_color[i][j] = bin_red[i][j][0:2] + bin_green[i][j][0:2] + bin_blue[i][j][0:1]

msb_color = np.zeros((rows, cols), dtype=int)
for i in range(0, rows):
	for j in range(0, cols):
		msb_color[i][j] = int(bin_color[i][j], 2)
		
plt.hist(msb_color.ravel(), 256, [0,256])
plt.title("color Channel")
plt.show()

plt.hist(msb_color.ravel(), 256, [0,256], normed=1)
plt.title("color Channel normalized")
plt.show()

plt.hist(msb_color.ravel(), 256, [0,256], normed=1, cumulative = -1)
plt.title("color Channel normalized and cumulative")
plt.show()

print("Median: ")
print(np.median(msb_color))

print("Variance: ")
print(np.var(msb_color))

print("Median: ")
print(np.median(msb_color))
#Compute and display cumulative histogram of an image. Hint: Use normalized histogram to obtain cumulative histogram.


cv.waitKey(0)
cv.destroyAllWindows()
