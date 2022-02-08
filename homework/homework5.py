import cv2 as cv
import numpy as np

img = cv.imread('../assets/shapesCorner.tif', cv.IMREAD_COLOR)
cv.imshow('orig', img)

rotate = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
gray = cv.cvtColor(rotate, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.07)
rotate[dst>0.01*dst.max()] = [0,0,255]
cv.imshow('rotate', rotate)

w = int(img.shape[1] * 60/100)
h = int(img.shape[0] * 60/100)
resize = cv.resize(img, (w,h), cv.INTER_AREA)
gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 5, 0.07)
resize[dst>0.01*dst.max()] = [0,0,255]
cv.imshow('resize', resize)


cv.waitKey(0)
cv.destroyAllWindows()
