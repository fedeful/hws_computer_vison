import cv2
import numpy as np





def prova(verbose=True):
    print('ciao')

    cap = cv2.VideoCapture('video_1.avi')
    index = 0
    fr1 = np.empty([])
    fr2 = np.empty([])
    while (cap.isOpened()):

        if index == 0:
            ret, fr1 = cap.read()
        elif index == 3:
            ret, fr2 = cap.read()
        elif index == 30:
            break
        else:
            ret, fr = cap.read()
        index = index + 1
    cap.release()

    if verbose:
        cv2.imshow('fr1', fr1)
        cv2.imshow('fr2', fr2)

        out = cv2.calcOpticalFlowFarneback(cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY),None,
                                           0.4, 1, 12, 2, 8, 1.2, 0)

        a = np.zeros((out.shape[0], out.shape[1], 3))
        a[..., 0] = out[..., 0]
        a[..., 1] = out[..., 1]
        cv2.imshow('out', a)

    if verbose:
        cv2.waitKey(0)


if __name__ == '__main__':

    prova()