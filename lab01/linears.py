from __future__ import print_function
import numpy as np
import cv2


def get_channels(img):
    if img.ndim == 3 and img.shape[2] > 1:
        return img.shape[2]
    return 1


def resize(img):
    return np.expand_dims(img, 2)


def saturate(a):
    if a > 255:
        return 255
    if a < 0:
        return 0
    return a


def negative(img, s, k):
    channels = get_channels(img)

    nimg = img
    if nimg.ndim == 2:
        nimg = resize(img)

    for ch in np.arange(0, channels):
        for row in np.arange(0, nimg.shape[0]):
            for column in np.arange(0, nimg.shape[1]):
                tmp = np.float64(0)
                tmp += s*nimg[row, column, ch] + k
                if tmp > 255:
                    tmp = 255
                if tmp < 0:
                    tmp = 0
                nimg[row, column, ch] = np.uint8(tmp)
    return nimg


def fast_negative(img, s, k):

    old_type = img.dtype
    img = img.astype(np.float64)
    img = img*s + k
    sa = np.vectorize(saturate)
    img = sa(img)
    img = img.astype(old_type)

    return img


def linear_blend(img1, img2, alpha):
    channels = get_channels(img1)

    nimg1 = cv2.resize(img1, (512, 512))
    if nimg1.ndim == 2:
        nimg1 = resize(nimg1)

    nimg2 = cv2.resize(img2, (512, 512))
    if nimg2.ndim == 2:
        nimg2 = resize(nimg2)

    blended = np.zeros(nimg1.shape,dtype=nimg1.dtype)
    for ch in np.arange(0, channels):
        for row in np.arange(0, blended.shape[0]):
            for column in np.arange(0, blended.shape[1]):
                tmp = np.float64(0)
                tmp += np.float64(1-alpha)*nimg1[row, column, ch] + np.float64(alpha)*nimg2[row, column, ch]
                if tmp > 255:
                    tmp = 255
                if tmp < 0:
                    tmp = 0
                blended[row, column, ch] = np.uint8(tmp)
    return blended


def fast_linear_blend(img1, img2, alpha):
    min1 = 0
    min2 = 0

    old_type = img1.dtype
    if img1.shape[0]>img2.shape[0]:
        min1 = img2.shape[0]
    else:
        min1 = img1.shape[0]

    if img1.shape[1]>img2.shape[1]:
        min2 = img2.shape[1]
    else:
        min2 = img1.shape[1]

    toup = (min2,min1)

    img1 = cv2.resize(img1, toup)
    img2 = cv2.resize(img2, toup)

    img1.astype(np.float64)
    img2.astype(np.float64)

    img1 = (1-alpha)*img1 + alpha*img2
    sa = np.vectorize(saturate)
    img1 = sa(img1)
    return img1.astype(old_type)


def contrast_stretching(img, new_max, new_min):
    channels = get_channels(img)

    if img.ndim == 2:
        img = resize(img)

    for k in np.arange(0, channels):
        ma = np.max(img[:, :, k])
        mi = np.min(img[:, :, k])
        for i in np.arange(0, img.shape[0]):
            for j in np.arange(0, img.shape[1]):
                pin = img[i, j, k]
                pin = float(pin) - mi
                pout = float(pin)*(float(new_max)-new_min)/(float(ma)-mi)
                pout += new_min
                print(pout)
                if pout < 0:
                    img[i, j, k] = 0
                elif pout > 255:
                    img[i, j, k] = 255
                else:
                    img[i, j, k] = np.uint8(pout)

    return img


def apply_threshold(img, threshold):

    direc = np.zeros(img.shape, dtype=np.uint8)
    inver = np.zeros(img.shape, dtype=np.uint8)

    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if img[i, j] < threshold:
                direc[i, j] = 0
                inver[i, j] = 255
            else:
                direc[i, j] = 255
                inver[i, j] = 0

    return direc, inver


def apply_mask_on_colored_image(img, mask):

    wback = np.zeros(img.shape, dtype=img.dtype)
    bback = np.zeros(img.shape, dtype=img.dtype)

    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            for k in np.arange(0, 3):
                if mask[i, j] == 0:
                    wback[i, j, k] = img[i, j, k]
                    bback[i, j, k] = img[i, j, k]
                else:
                    wback[i, j, k] = 255
                    bback[i, j, k] = 0
    return wback, bback