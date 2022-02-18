import cv2 as cv
import numpy as np

img = cv.imread("../assets/cameraman.tif", cv.IMREAD_GRAYSCALE)
t = cv.imread("../assets/cameramantemp.png", cv.IMREAD_GRAYSCALE)

w, h = t.shape[::-1]

####part a
img_cross_corr = img.copy()
img_cross_corr_norm = img.copy()

res1 = cv.matchTemplate(img, t, cv.TM_CCORR_NORMED)
res2 = cv.matchTemplate(img, t, cv.TM_CCORR)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)

cv.rectangle(img_cross_corr_norm, top_left, bottom_right, 255, 2)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res2)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.rectangle(img_cross_corr, top_left, bottom_right, 255, 2)
    
#cv.imshow("cross correlation normed", img_cross_corr_norm)
#cv.imshow("cross correlation ", img_cross_corr)


###part b
kernel = np.array([
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]
])
t_intensity = cv.filter2D(t, -1, kernel)
img_cross_corr = img.copy()
img_cross_corr_norm = img.copy()

res1 = cv.matchTemplate(img, t_intensity, cv.TM_CCORR_NORMED)
res2 = cv.matchTemplate(img, t_intensity, cv.TM_CCORR)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)

cv.rectangle(img_cross_corr_norm, top_left, bottom_right, 255, 2)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res2)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.rectangle(img_cross_corr, top_left, bottom_right, 255, 2)
#cv.imshow("cross correlation normed", img_cross_corr_norm)
#cv.imshow("cross correlation ", img_cross_corr)


##### Part C
img_intensity = cv.filter2D(img, -1, kernel)
img_cross_corr = img_intensity.copy()
img_cross_corr_norm = img_intensity.copy()

res1 = cv.matchTemplate(img_intensity, t, cv.TM_CCORR_NORMED)
res2 = cv.matchTemplate(img_intensity, t, cv.TM_CCORR)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)

cv.rectangle(img_cross_corr_norm, top_left, bottom_right, 255, 2)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res2)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.rectangle(img_cross_corr, top_left, bottom_right, 255, 2)
#cv.imshow("cross correlation normed", img_cross_corr_norm)
#cv.imshow("cross correlation ", img_cross_corr)


#######PART D
img_cross_corr = img_intensity.copy()
img_cross_corr_norm = img_intensity.copy()

res1 = cv.matchTemplate(img_intensity, t_intensity, cv.TM_CCORR_NORMED)
res2 = cv.matchTemplate(img_intensity, t_intensity, cv.TM_CCORR)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)

cv.rectangle(img_cross_corr_norm, top_left, bottom_right, 255, 2)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res2)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.rectangle(img_cross_corr, top_left, bottom_right, 255, 2)
#cv.imshow("cross correlation normed", img_cross_corr_norm)
#cv.imshow("cross correlation ", img_cross_corr)

########PART E
t_rot = cv.rotate(t, cv.ROTATE_90_CLOCKWISE)
img_cross_corr = img_intensity.copy()
img_cross_corr_norm = img_intensity.copy()

res1 = cv.matchTemplate(img, t_rot, cv.TM_CCORR_NORMED)
res2 = cv.matchTemplate(img, t_rot, cv.TM_CCORR)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)

cv.rectangle(img_cross_corr_norm, top_left, bottom_right, 255, 2)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res2)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1]+h)
cv.imshow("cross correlation normed", img_cross_corr_norm)
cv.imshow("cross correlation ", img_cross_corr)

cv.waitKey(0)
cv.destroyAllWindows()
