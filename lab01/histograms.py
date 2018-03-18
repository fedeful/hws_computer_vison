from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def n_list_generator(value, times):
    return [value]*times


def histogram(img):
    h = n_list_generator(0, 256)
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            h[img[i, j]] += 1
    return h


def normalized_histogram(h):

    if type(h) == np.ndarray:
        h = histogram(h)

    pixels = np.sum(h)
    n_hist = np.divide(h, np.float(pixels))

    return n_hist


def print_histogram(histogram, title='Histogram'):
    if type(histogram) == list:
        plt.bar([i for i in np.arange(0, 256)], histogram, 1)
        plt.title(title)
        plt.show()
        return 0
    return -1


def prior_prob_range(norm_hist, beg, end):
    w = 0.
    for i in np.arange(beg, end):
        w += norm_hist[i]
    return w


def avg_range(norm_hist, beg, end):
    m = 0.
    for i in np.arange(beg, end):
        tmp = i*np.float(norm_hist[i])
        m += tmp
    w = prior_prob_range(norm_hist, beg, end)

    if w == 0:
        return 0

    m = m/w
    return m


def var_range_hist(norm_hist, beg, end):
    var = 0.
    avg = avg_range(norm_hist, beg, end)
    for i in np.arange(beg, end):
        tmp = np.square((i-avg))*np.float(norm_hist[i])
        var += tmp
    if prior_prob_range(norm_hist, beg, end) == 0:
        return 0
    var = var/prior_prob_range(norm_hist, beg, end)
    return var


def inter_class_variance(norm_hist, th):
    w1 = prior_prob_range(norm_hist, 0, th+1)
    w2 = prior_prob_range(norm_hist, th+1, 256)
    mou1 = avg_range(norm_hist, 0, th+1)
    mou2 = avg_range(norm_hist, th+1, 256)
    return w1*w2*np.power((mou1-mou2), 2)


def find_threshold(norm_hist):
    best_number = 0
    best_value = 0.
    x = []
    for i in np.arange(0, 256):
        var = inter_class_variance(norm_hist, i)
        x.append(var)
        if var > best_value:
            best_value = var
            best_number = i
    print(x)
    return best_number


def otsu(image):
    thresholds = []
    if image.ndim == 3:
        for i in np.arange(0, image.shape[2]):
            h = histogram(image[:, :, i])
            h = normalized_histogram(h)
            thresholds.append(find_threshold(h))
    elif image.ndim == 2:
        h = histogram(image)
        h = normalized_histogram(h)
        thresholds.append(find_threshold(h))
    return thresholds





