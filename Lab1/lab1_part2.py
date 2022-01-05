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


