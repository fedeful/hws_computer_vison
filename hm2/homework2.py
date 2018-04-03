from __future__ import print_function
from lab01.linears import contrast_stretching
import numpy as np
import matplotlib.pyplot as plt
import cv2


def contrast_starching2():
    caffe = cv2.imread('./img/aeBef_constr_stre.jpg', cv2.IMREAD_COLOR)
    eq = np.zeros(caffe.shape, dtype=caffe.dtype)
    eq[:, :, 0] = cv2.equalizeHist(caffe[:, :, 0])
    eq[:, :, 1] = cv2.equalizeHist(caffe[:, :, 1])
    eq[:, :, 2] = cv2.equalizeHist(caffe[:, :, 2])

    eqim = np.hstack((caffe, eq))
    cv2.imwrite('./img_processed/aeafter_const_stre.jpg', eq)
    cv2.imshow('eq', eqim)
    cv2.waitKey(0)


def add_gaussian_noise_to_image(img, mean=0, sig=10):
    if np.ndim(img) != 3:
        np.expand_dims(img, 2)

    gauss = np.random.normal(0, sig, img.shape)
    noisy_im = img + gauss

    return noisy_im


def gauss_on_noised_image(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return blur


def add_sp_noise_to_image(img, spratio, per_sub_pix, seed=46):
    if np.ndim(img) != 3:
        np.expand_dims(img, 2)

    np.random.seed(seed)
    spn = int(spratio*img.shape[0]*img.shape[1]*per_sub_pix)
    ppn = int((1-spratio) * img.shape[0] * img.shape[1] * per_sub_pix)

    sp = [np.random.randint(0, i - 1, spn) for i in img.shape[0:2]]
    pp = [np.random.randint(0, i - 1, ppn) for i in img.shape[0:2]]

    img[sp] = np.array([0, 0, 0], dtype=np.uint8)
    img[pp] = np.array([255, 255, 255], dtype=np.uint8)

    return img


def adaptive_threshold(infile, outfile, colored=False):
    im = cv2.imread('../lab02/img/sonnet.jpg', cv2.IMREAD_GRAYSCALE)
    ret = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
    cv2.imshow('eq', ret)
    cv2.waitKey(0)
    cv2.waitKey(0)


def otzu_thresh(img):
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def bin_thre(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh1


def adaptative_thr(img):
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 9)
    return th3


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return x, y


def canny(img):
    im = cv2.Canny(img, 50, 150)
    return im


def Dog(img):
    blur5 = cv2.GaussianBlur(img, (7, 7), 0)
    blur3 = cv2.GaussianBlur(img, (3, 3), 0)

    # write the results of the previous step to new files


    return blur5 - blur3



if __name__ == '__main__':
    im = cv2.imread('./img/shing.jpg', cv2.IMREAD_COLOR)
    im = Dog(im)
    #plt.hist(im.ravel(), 256, [0, 256])
    #plt.show()
    #im2 = contrast_stretching(im,150,0)
    #im = add_sp_noise_to_image(im,0.5,0.01,465423)
    #im = add_gaussian_noise_to_image(im, 0, 60)
    #im = cv2.medianBlur(im, 5)
    #plt.hist(im.ravel(), 256, [0, 256])
    #plt.show()
    #plt.hist(im2.ravel(), 256, [0, 256])
    #plt.show()
    #m = otzu_thresh(im)
    #cv2.imshow('cos',im2)
    #cv2.waitKey(0)

    #im = cv2.bilateralFilter(im, 11, 100, 100)
    #kernel = np.ones((21, 21   ), np.float32) / 441
    #dst = cv2.filter2D(im, -1, kernel)


    cv2.imwrite('./img_processed/shing2.jpg', im)

