import numpy as np
import cv2

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == '__main__':
    a = cv2.imread('/home/fede/Desktop/img/modena_skyline_49.png')
    cv2.imshow('pippo', a)

    a2 = rotateImage(a, 70)
    cv2.imshow('pippo2', a2)
    cv2.waitKey(0)

