# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


class Imb():

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    def get_image_with_box(self, img):

        orig = img.copy()

        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=0.95)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        # filename = imagePath[imagePath.rfind("/") + 1:]
        #print("[INFO] {}: {} original boxes, {} after suppression".format(
        #    filename, len(rects), len(pick)))

        return img


if __name__ == '__main__':
    imb = Imb()
    ret = imb.get_image_with_box(cv2.imread('/home/fede/PycharmProjects/computer_vision/hm5/image/aa.jpg'))

    cv2.imshow('prova', ret)
    cv2.waitKey(0)