from __future__ import print_function
from os.path import join,abspath,isdir,isfile
from os import getcwd,listdir, chdir
import cv2
import numpy as np
import PIL.Image


chdir('/home/fede/Desktop/Label')

for i in listdir(getcwd()):
    if isfile(i):
        im =cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        im[im == 0] = 255
        im[im == 1] = 0
        im[im == 2] = 128
        cv2.imwrite(i, im)
