# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab03/img/modena_skyline_05.png', cv2.IMREAD_COLOR).astype(np.float64)
#image = img_as_float(io.imread(args["image"]))

# loop over the number of segments
for numSegments in (50, 70,100,150):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments=numSegments, sigma=5)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image.astype(np.uint8), segments))
    plt.axis("off")

# show the plots
plt.show()