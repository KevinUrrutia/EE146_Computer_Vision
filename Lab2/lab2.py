import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#import and read the image, display it as well as its histogram
gray_img = cv.imread("../assets/non-textured.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("grayed image", gray_img)

#use opencv algorithm to segment the image
otsu_threshold, image_result = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("image_result", image_result)


#calculate the histogram of the image
hist, bin_edges = np.histogram(gray_img, bins=256, density=False)

#create arrays to store the foreground var, background var, inter var 
fground_var = np.zeros([256,1])
bground_var = np.zeros([256,1])
inter_var = np.zeros([256,1])

print(hist)
#create a 1D sum array
var_sum = np.zeros([255,1])

#for threshold from 0 to 255
for t in range(0, 255):
	#binarize the image based on the threshold
	fground = hist[:t+1]
	bground = hist[t+1:]
	
	#find the varience of the foreground and background
	fground_var[t] = np.var(fground)
	bground_var[t] = np.var(bground)
	
	#find the varience between the two variences
	#var_array = np.array([fground_var[t], bground_var[t]])
	#inter_var[t] = np.var(var_array)
	q1 = np.cumsum(fground)
	q2 = np.cumsum(bground)
	mu1 = np.mean(fground)
	mu2 = np.mean(bground)
	inter_var[t] = q1[0] * q2[0] * (mu1 - mu2) ** 2

	
	#sum the varience of the forground and background and store into a 1d array
	var_sum[t] = fground_var[t] + bground_var[t] 

#find the maximum of inter_var
thresh_max = np.where(inter_var == np.amax(inter_var))
print(thresh_max)
#scan the sum array and find its minimum
min_element = np.amin(var_sum)
#find the threshold of the minimum value
thresh = np.where(var_sum == np.amin(var_sum))
print(thresh)

gray_img[gray_img>thresh[0]] = 255
gray_img[gray_img!=255] = 0

fground_var.reshape([1, 256])
bground_var.reshape([1, 256])
inter_var.reshape([1, 256])

#plot the variences and the histogram
t = np.linspace(0, 256, num=256)
plt.plot(t, hist, 'b')
plt.title("histogram")
plt.show()

plt.plot(t, fground_var, 'r')
plt.plot(t, bground_var, 'g')
plt.plot(t, inter_var, 'k')
plt.legend(["foreground var", "background var", "inter var"])
plt.show()

cv.imshow("threshold_img", gray_img)
#threshold the image at the computed threshold
	
cv.waitKey(0)
cv.destroyAllWindows()

