from __future__ import print_function
import cv2
import numpy as np
import math
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':
    modena_img = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab03/img/modena_skyline_07.png', cv2.IMREAD_COLOR)
    modena_orig = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab03/img/modena_skyline_07.png', cv2.IMREAD_COLOR)

    case = 2
    if case == 1:
        modena_img = cv2.cvtColor(modena_img, cv2.COLOR_BGR2GRAY)
        modena_img = cv2.bilateralFilter(modena_img, 5, 75, 75)
        modena_img = cv2.Canny(modena_img, 150, 250)
    elif case == 2:

        modena_img = cv2.cvtColor(modena_img, cv2.COLOR_BGR2GRAY)
        modena_img = cv2.bilateralFilter(modena_img, 11, 100, 100)
        ret, th1 = cv2.threshold(modena_img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('prov', th1)

        cv2.imshow('before canny', modena_img)
        modena_img = cv2.Canny(modena_img, 200, 240)


    lines = cv2.HoughLinesP(modena_img, 1, np.pi / 180, 70, None, 70,130)
    disimg = np.zeros(modena_orig.shape, dtype=modena_orig.dtype)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(disimg, pt1, pt2, (0, 0, 255),2, cv2.LINE_AA)

    h = modena_orig.shape[0]
    w = modena_orig.shape[1]
    for i in np.arange(0, h):
        for j in np.arange(0, w):
            if np.array_equal(disimg[i, j, :], [0, 0, 255]):
                modena_orig[i, j, :] = np.array([0, 0, 255])

    cv2.imshow('edge', modena_img)
    cv2.imshow('line', modena_orig)
    k = cv2.waitKey(0)
    if k == 'q':
        cv2.destroyAllWindows()
    print('ciao')

