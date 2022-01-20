import cv2 as cv
import numpy as np


def getNeighbors(I, u, v): #assuming 4 connectivity
	N = np.zeros([2, 1])
	if(u == 0):
		N[0] == 0
	else:
		N[0] = I[u-1, v]
		N[1] = I[u, v-1]
	return N
	

def sequentialLabeling(I):
	'''
	Input: I, an integer-valued image with initial values 
	0 = background, 1 = foreground. Returns nothing by modifies image I
	'''
	(M, N) = I.shape
	label = 2 # value of the next label to assigned
	C = {}	#creates empty list of label collsion
	#print(I)

	#first pass
	for v in range(0, N):
		for u in range(0, M):	
			if(I[u][v] == 1):
				N = getNeighbors(I, u, v)
				if((N[0] and N[1]) == 0):
					I[u][v] = label
					label = label + 1
				elif((N[0] > 1) ^ (N[1] > 1)):
					I[u][v] = N[0]
				elif((N[0] > 1) and (N[1] > 1)):
					N[0]=N[0].astype(int)
					N[1]=N[1].astype(int)
					I[u][v] = N[0]
					C.update({int(N[0]): int(N[1])})
	
	#relabel image
	R = {}
	for i in range(2, label):
		R.update({i: i})
	
	for k, v in C.items():
		print(v)
		R[k] = R[k] + v
	
	print(R)
	return I

'''
img = cv.imread("../assets/connected_components.png", cv.IMREAD_GRAYSCALE)
otsu_thresh, bin_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

cv.imshow("bin_img", bin_img)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
erode = cv.erode(bin_img, kernel, iterations=1)
dilate = cv.dilate(erode, kernel, iterations=3)
erode = cv.erode(dilate, kernel, iterations=1)

cv.imshow("morph_img", erode)
'''
img = np.array([[0, 0, 1, 0, 0, 1, 1, 1], 
		[0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0],
		[1, 1, 1, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 0, 1, 1, 1]])

connected_img = sequentialLabeling(img)

cv.waitKey(0)
cv.destroyAllWindows()
