from __future__ import print_function
from histograms import otsu
from linears import apply_mask_on_colored_image, apply_threshold
import numpy as np
import cv2


def prove_con_otzu(filename):
    o = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    th = otsu(o)
    direc, inver = apply_threshold(o, th)
    cv2.imshow('diretta', direc)
    cv2.imshow('inversa', inver)

    o = cv2.imread(filename, cv2.IMREAD_COLOR)
    dwback, dbback = apply_mask_on_colored_image(o, direc)
    iwback, ibback = apply_mask_on_colored_image(o, inver)

    cv2.imshow('diretta sfondo bianco', dwback)
    cv2.imshow('diretta sfondo nero', dbback)

    qu = cv2.waitKey(0)
    if qu == 'q':
        cv2.destroyAllWindows()


def adaptive_thresholding(img, ws, c):
    for i in np.arange(0, img.shape[0], ws):
        for j in np.arange(0, img.shape[1], ws):
            m = img[i:i+ws, j:j+ws]
            wm = np.mean(m)
            row = ws
            col = ws
            if i + ws >= img.shape[0]:
                row = img.shape[0] - i
            if j + ws >= img.shape[1]:
                col = img.shape[1] - j
            for k in np.arange(0, row):
                for l in np.arange(0, col):
                    if img[i+k, j+l] < (wm - c):
                        img[i + k, j + l] = 0
                    else:
                        img[i + k, j + l] = 255
    return img


if __name__ == '__main__':
    a = cv2.imread('../lab02/img/sonnet.jpg', cv2.IMREAD_GRAYSCALE)
    a = adaptive_thresholding(a, 17, 10)
    cv2.imshow('adaptive thres', a)

    qu = cv2.waitKey(0)
    if qu == 'q':
        cv2.destroyAllWindows()
