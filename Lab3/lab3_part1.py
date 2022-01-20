import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def dilate(I, H):
	'''
	Input: I, a binary image size MxN
	H, a binary structuring element
	Returns the Dialated image I' = I or H
	'''
	#create a new binary image I'
	print(I.shape)
	I_prime = np.zeros((I.shape[0], I.shape[1]))


	for i in range(int(math.ceil(H.shape[0] /2)), I.shape[0] - int(math.floor(H.shape[1]/2))): #map I into the kernel H in both x and y directions
		for j in range(int(math.ceil(H.shape[1] /2)), I.shape[1] - int(math.floor(H.shape[1]/2))):
	
			#get the nieghbors of current pixels of I
			neigh = I[i-int(math.floor(H.shape[0]/2)):i+int(math.floor(H.shape[0]/2)), j-int(math.floor(H.shape[1]/2)):j+int(math.floor(H.shape[1]/2))]
			
			#use 0 and 1 instead of 0 and 255
			if(255 in neigh):
				bin_neigh = neigh /255
			else:
				bin_neigh = neigh
			
			#in the new map place 1 in all locations in where 1 was found in the neighbors using the structuring element
			I_prime[i][j] = np.max(bin_neigh)
	
	return I_prime

def erode(I, H):
	
	#create a new binary image I'
	print(I.shape)
	I_prime = np.zeros((I.shape[0], I.shape[1]))
	print(H)

	for i in range(int(math.ceil(H.shape[0] /2)), I.shape[0] - int(math.floor(H.shape[1]/2))): #map I into the kernel H in both x and y directions
		for j in range(int(math.ceil(H.shape[1] /2)), I.shape[1] - int(math.floor(H.shape[1]/2))):
	
			#get the nieghbors of current pixels of I
			neigh = I[i-int(math.floor(H.shape[0]/2)):i+int(math.floor(H.shape[0]/2)), j-int(math.floor(H.shape[1]/2)):j+int(math.floor(H.shape[1]/2))]
			
			#use 0 and 1 instead of 0 and 255
			neigh = neigh.astype(int)
			print(neigh)
			if(255 in neigh):
				bin_neigh = neigh /255
			else:
				bin_neigh = neigh
			#print(type(bin_neigh))
			
			#in the new map place 1 in all locations in where 1 was found in the neighbors using the structuring element
			I_prime[i][j] = np.min(bin_neigh)
	
	return I_prime
	

img = cv.imread("../assets/morph_image.png", cv.IMREAD_GRAYSCALE)
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
cv.imshow("orig", bin_img)

#create a structuring element whose size is 5 x 5 pixels
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

#dilate the image
dilate_img = dilate(bin_img, kernel) 
cv.imshow("dilated image", dilate_img)

eroded_img = erode(bin_img, kernel)
cv.imshow("eroded image", eroded_img)

cv.waitKey(0)
cv.destroyAllWindows()
