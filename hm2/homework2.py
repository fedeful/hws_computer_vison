from __future__ import print_function
import numpy as np
import cv2


def contrast_starching():
    caffe = cv2.imread('./img/lowcon.jpg', cv2.IMREAD_COLOR)
    eq = np.zeros(caffe.shape, dtype=caffe.dtype)
    eq[:, :, 0] = cv2.equalizeHist(caffe[:, :, 0])
    eq[:, :, 1] = cv2.equalizeHist(caffe[:, :, 1])
    eq[:, :, 2] = cv2.equalizeHist(caffe[:, :, 2])

    eqim = np.hstack((caffe, eq))
    cv2.imshow('eq', eqim)
    cv2.waitKey(0)


def add_gaussian_noise_to_image(img, mean=0, sig=10):
    if np.ndim(img) != 3:
        np.expand_dims(img, 2)

    gauss = np.random.normal(0, sig, im.shape)
    noisy_im = im + gauss

    return noisy_im


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


if __name__ == '__main__':
    im = cv2.imread('./img/meres.jpg', cv2.IMREAD_COLOR)
    #im = add_sp_noise_to_image(im,0.5,0.01,465423)
    #im = add_gaussian_noise_to_image(im, 0, 60)


    cv2.imwrite('./img/megausnoise.jpg', im)
