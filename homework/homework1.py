'''
Input: I, a scaler-valued input image with I(u,v) in the real number space.
Output: Mean and variance of a histogram
'''

#read in the image
img = cv.imread("../", cv.IMREAD_GRAYSCALE)

#Get the size of the image and save this information as rows and cols
rows = img.shape[0]
cols =  img.shape[1]

#Get the total values of the image I(u,v), as well as the square of the values I^2(u,v)
total_val = 0
sqr_total_val = 0
for i in range(0,rows):
    for j in range(0,cols):
        total_val += img[i][j]
        sqr_total_val += 2 ** img[i][j]
#Calculate Mean
mean = (total_val) / (rows * cols)

#calculate Variance
var = (sqr_total_val - ((rows*cols)(2 ** total_val))) * (rows * cols)
return var and mean

'''
Input: I, a scaler-valued input image with I(u,v) in the real number space.
Output: Thresheld image I_thresh
'''
#read in the image
img = cv.imread("../", cv.IMREAD_GRAYSCALE)

#Find the min and max values of the image
max = np.amax(img)
min = np.amin(img)

#get the median
median = (max + min) / 2

#if the image pixel value is above median then it becomes 255, if below it is zero
rows = img.shape[0]
cols =  img.shape[1]
for i in range(0,rows):
    for j in range(0,cols):
        if(img[i][j] => median):
            img[i][j] = 255
        else:
            img[i][j] = 0


'''
Input: I, a scaler-valued input image with I(u,v) in the real number space.
Output: Mean and stadard deviation
'''
#Get the size of the image and save this information as rows and cols
rows = img.shape[0]
cols =  img.shape[1]

#Get the total values of the image I(u,v), as well as the square of the values I^2(u,v)
total_val = 0
sqr_total_val = 0
for i in range(0,rows):
    for j in range(0,cols):
        total_val += img[i][j]

#Calculate Mean
mean = (total_val) / (rows * cols)

#calculate Variance
var = (sqr_total_val - ((rows*cols)(2 ** total_val))) * (rows * cols)

#calculate standard deviation
SD = sqr(var)
return SD and mean
