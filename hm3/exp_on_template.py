from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt





img = cv2.imread('./template/daytemp.png')
cv2.imshow('ghir', img)
a = img[60:90, 14:30]
cv2.imshow('ghir1',a)

a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
cv2.imshow('ghir2',a)

#plt.hist(img[:,:,2].ravel(),256,[0,256]); plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 0, 0])
upper_red = np.array([170, 55, 160])

#lower_red = np.array([a[:,:,0].min(), a[:,:,1].min(), a[:,:,2].min()])
#upper_red = np.array([a[:,:,0].max(), a[:,:,1].max(), a[:,:,2].max()])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask=mask)
#blur = cv2.Filter(res,(3,3),0)
closing =  cv2.dilate(res,None,iterations=2)
closing = cv2.erode(closing,None, iterations=3)
#closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, (7, 7))
bff = cv2.bilateralFilter(closing,4,70,70)
cv2.imshow('prova',res)
cv2.imshow('prova2',bff)

cv2.waitKey(0)