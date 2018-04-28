import cv2
import numpy as np

tmp_img = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab01/img/lenac.png')
cv2.imshow('prav1', tmp_img)

prev_shape = tmp_img.shape

tmp_img = cv2.resize(tmp_img, (150, 150))
cv2.imshow('prav2', tmp_img)

tmp_img = cv2.resize(tmp_img, prev_shape[:2])
cv2.imshow('prav3', tmp_img)


cv2.waitKey(0)
