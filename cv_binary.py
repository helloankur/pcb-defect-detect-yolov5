import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


img=cv2.imread('pcb1.jpg')



img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#thresh = cv2.threshold(thresh, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRUNC)[1]


cv2.imshow('1',thresh)
cv2.waitKey()


