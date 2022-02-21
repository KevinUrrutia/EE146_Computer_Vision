import cv2 as cv
import numpy as np
import scipy.ndimage
import time

def chamfer_match(img, t):
    [m, n] = img.shape
    [r, c] = t.shape
    
    
    D = scipy.ndimage.distance_transform_edt(img)
    Q = np.zeros((m-r, n-c))
    
    
    fg = np.count_nonzero(t)
    for i in range(1, m-r):
        for j in range(1, n-c):
            tempI = D[i:i+r-1, j:j+c-1]
            Q[i,j] = np.sum(tempI) / fg
    return Q
            
img = cv.imread("../assets/BW.png", cv.IMREAD_GRAYSCALE)
t = cv.imread("../assets/R.png", cv.IMREAD_GRAYSCALE)


###PART A
w, h = t.shape[::-1]

img_cross_corr = img.copy()
start = time.time()
res = cv.matchTemplate(img, t, cv.TM_CCORR)
end = time.time()
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.rectangle(img_cross_corr, top_left, bottom_right, 255, 1)
cross_corr_time = end - start

print(f"Cross Correlation time: {cross_corr_time}")
cv.imshow("cross_corr", img_cross_corr)

#####PARTB
start = time.time()
Q = chamfer_match(img, t)
min_loc = np.where(Q == np.amin(Q))
end = time.time()
for i in range(min_loc[0].size):
    cv.rectangle(img, (min_loc[0][i],min_loc[1][i]), (min_loc[0][i]+w, min_loc[1][i]+h), 255, 1)
print(f"Chamfer match time: {end-start}")
cv.imshow('chamfer', img)

cv.waitKey(0)
cv.destroyAllWindows()
