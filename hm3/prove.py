from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
im = cv2.imread('./template/daytemp.png', cv2.IMREAD_GRAYSCALE)
closing = cv2.morphologyEx(im, cv2.MORPH_DILATE, (3, 3))
closing = cv2.morphologyEx(closing, cv2.MORPH_ERODE, (3, 3))
bff = cv2.bilateralFilter(closing,3,50,50)


cv2.imshow('prova2', closing)
o = cv2.Canny(closing, 160, 240)

cv2.imshow('sac', o)

im2, contours, hierarchy = cv2.findContours(o, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.watershed(im2,contours)
watershed( im, markers );
#cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
#a =cv2.floodFill(cv2.cvtColor(im,cv2.COLOR_GRAY2BGR), contours, 255,255)

#cv2.imshow('prova', a)
cv2.waitKey(0)
#cv2.fillPoly()
'''
import sys
import cv2
import numpy
from scipy.ndimage import label

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=1)
    border = border - cv2.erode(border, None, iterations=2)

    dt = cv2.distanceTransform(img, 2, 3)
    cv2.imshow('prova', dt)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


img = cv2.imread('./template/daytemp.png')

bf1 = cv2.dilate(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), None, iterations=1)

bf2 = cv2.erode(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), None, iterations=2)
cv2.imshow('bil',bf1 -bf2)
# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=int))
cv2.imshow('bin', img_bin)
result = segment_on_dt(img, img_bin)
cv2.imshow('prova1', result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
cv2.imshow('prova2', img)
cv2.waitKey(0)