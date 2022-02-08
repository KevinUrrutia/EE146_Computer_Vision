'''
import cv2 as cv
import numpy as np
from scipy.ndimage import convolve


I = np.array([[14, 10, 19, 16, 14, 12],
              [18, 9, 11, 12, 10, 19],
              [9, 14, 15, 26, 13, 6],
              [21, 27, 17, 17, 19, 16],
              [11, 18, 18, 19, 16, 14],
              [16, 10, 13, 7, 22, 21]])

print(I)
              
H = np.array([[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]])

print(H)
             
I_prime = convolve(I, H)

print(I_prime)
'''

import cv2 as cv

img = ("../assets/median_filter.png", cv.IMREAD, GRAYSCALE)

median_filter = cv.medianBlur(img, 5)
