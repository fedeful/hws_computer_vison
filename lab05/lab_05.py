from __future__ import print_function
import numpy as np
import cv2


def main_app(pmode=True):
    img1 = cv2.imread('./stichingf/Foto424.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('./stichingf/Foto425.jpg', cv2.IMREAD_COLOR)

    img1 = cv2.resize(img1, (int(img1.shape[1]*0.25), int(img1.shape[0]*0.25)))
    img2 = cv2.resize(img2, (int(img2.shape[1] * 0.25), int(img2.shape[0] * 0.25)))

    if pmode:
        cv2.imshow('prima foto', img1)
        cv2.imshow('seconda foto', img2)

    h, status = cv2.findHomography(np.array([[308, 303], [83, 435], [423, 472], [87, 362]]),
                                   np.array([[608, 303], [383, 435], [723, 472], [387, 362]]))

    im_dst1 = cv2.warpPerspective(img1, h, (img1.shape[1]*2, img1.shape[0]*2))

    h2, status2 = cv2.findHomography(np.array([[430, 23], [216, 166], [552, 187], [214, 92]]),
                                   np.array([[608, 303], [383, 435], [723, 472], [387, 362]]))

    im_dst2 = cv2.warpPerspective(img2, h2, (img1.shape[1] * 2, img1.shape[0] * 2))

    if pmode:
        cv2.imshow('prima foto 2', im_dst1)
        cv2.imshow('seconda foto 2', im_dst2)

    imgf = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2, 3), dtype=np.uint8)

    imgf[im_dst1[:, :] != [0, 0, 0]] = im_dst1[im_dst1[:, :] != [0, 0, 0]]
    imgf[imgf[:, :] == [0, 0, 0]] = im_dst2[imgf[:, :] == [0, 0, 0]]

    if pmode:
        cv2.imshow('fianle', imgf)

    if pmode:
        cv2.waitKey(0)


def hough_prove(pmode=True):
    img1 = cv2.imread('./houghf/6.BMP', cv2.IMREAD_GRAYSCALE)

    if pmode:
        cv2.imshow('prima foto', img1)

    edges = cv2.Canny(img1, 100, 200)

    if pmode:
        cv2.imshow('immagine con i bordi', edges)

    if pmode:
        cv2.waitKey(0)


if __name__ == '__main__':
    #hough_prove()
    main_app()