from __future__ import print_function
import numpy as np


def colored_function(appfunc, img, *args):
    for ch in np.arange(0, 3):
        img[:, :, ch] = appfunc(img[:, :, ch], *args)
    return img