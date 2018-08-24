from __future__ import print_function
import numpy as np
import cv2
# le operazioni vanno castate a float altrimenti sforo i byte
# poi devo riportarlo tra 0 e 255


Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)*(1/8.)
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)*(1/8.)
lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

def gradients(img, kernel):
    grad = cv2.filter2D(img, -1, kernel)
    return grad


def magnitude(Gx, Gy):
    tmp = (Gx**2 + Gy**2)**0.5
    norm = ((255.*3/8)**2+(255.*3/8)**2)**0.5

    return (tmp/norm)*255


def theta(Gx, Gy):
    return np.arctan2(Gy, Gx)

def sobel():
    print(Sx)
    print(Sy)
    modena_skyline = cv2.imread('./img/modena_skyline_03.png',
                                cv2.IMREAD_GRAYSCALE)
    modena_skyline = modena_skyline.astype(dtype=np.float64)

    gradx = gradients(modena_skyline, Sx)
    grady = gradients(modena_skyline, Sy)
    # cv2.imshow('magnite', gradx.astype(np.uint8))
    # cv2.imshow('magnitey', grady.astype(np.uint8))

    H = theta(gradx, grady)
    V = magnitude(gradx, grady)
    S = np.ones(V.shape, dtype=np.float64) * 255
    # hsv = np.array([H, S, V])
    H = (180. - 0) / (np.amax(H) - np.amin(H)) * (H - np.amin(H))
    im = np.zeros((H.shape[0], H.shape[1], 3), dtype=np.float64)
    im[:, :, 0] = H
    im[:, :, 1] = S
    im[:, :, 2] = V
    hsv = im
    hsv = hsv.astype(dtype=np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('magnitude', hsv)
    cv2.waitKey(0)


def log():
    LoG = cv2.imread('./img/modena_skyline_03.png', cv2.IMREAD_GRAYSCALE)
    LoG = LoG.astype(dtype=np.float64)
    LoG = cv2.filter2D(LoG, -1, lap)
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3, 3)))
    T = 60
    LoG = np.logical_and(LoG < -T, maxLoG > T) * 255

    cv2.imshow('log', LoG.astype(np.uint8))
    cv2.waitKey(0)


def canny():
    a = cv2.imread('./img/modena_skyline_03.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(a, 100, 200)
    cv2.imshow('canny', edges)
    cv2.waitKey(0)


if __name__ == '__main__':

    kind = 'canny'

    if kind == 'sobel':
        print('sobel')
        sobel()
    elif kind == 'log':
        print('log')
        log()
    elif kind == 'canny':
        print('canny')
        canny()
