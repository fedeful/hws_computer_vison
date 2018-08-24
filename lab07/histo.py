import numpy as np
import cv2
import matplotlib.pyplot as plt


def print_histogram(histogram, title='Histogram'):
    plt.bar([i for i in np.arange(0, len(histogram))], histogram, 1)
    plt.title(title)
    plt.show()


def histogram(img, nbin=256, norm=True):
    h = [0]*nbin
    tmp = np.resize(img, (-1))
    for i in np.arange(0, len(tmp)):
        h[int((tmp[i]/float(256))*nbin)] += 1
    if norm:
        h = np.divide(h, float(len(tmp)))



    return h


def bgr_hist(img, nbin=256, norm=True):
    tot_hist = np.zeros([], dtype=np.uint8)
    for i in np.arange(0, 3):
        if i == 0:
            tot_hist = histogram(img[:, :, i], nbin)
        else:
            tot_hist = np.concatenate((tot_hist, histogram(img[:, :, i], nbin)), axis=0)

    return tot_hist


def joint_hist(img, nbin=256, norm=True):
    h = np.zeros((nbin, nbin, nbin))
    tmp = np.resize(img, (-1, 3))
    print(len(tmp))
    for i in np.arange(0, len(tmp)):

        h[(tmp[i]/float(256)*nbin).astype(np.int)] += 1
    if norm:
        h = np.divide(h, float(len(tmp)))

    return h


def main(verb=True):
    img = cv2.imread('mountain.jpg', cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (128, 128))

    if verb:
        cv2.imshow('immagine', img)



    #h = bgr_hist(img)
    h = joint_hist(img)

    #print_histogram(h)

    if verb:
        cv2.waitKey(0)


if __name__ == '__main__':
    main()