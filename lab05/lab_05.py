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
    img1 = cv2.imread('./houghf/12.BMP', cv2.IMREAD_GRAYSCALE)
    img2 = np.expand_dims(img1, 2)
    img2 = np.repeat(img2, 3, 2)
    if pmode:
        cv2.imshow('prima foto', img1)

    edges = cv2.Canny(img1, 100, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 50)

    if lines is not None:

        for i in np.arange(0, len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # if (max_x - 50) > x1 > tmp_max:
                #    tmp_max = x1
                cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if pmode:
        cv2.imshow('immagine con i bordi', img2)

    if pmode:
        cv2.waitKey(0)


if __name__ == '__main__':

    kind = 'hough'
    if kind == 'hough':
        hough_prove()
    elif kind == 'stich':
        main_app()