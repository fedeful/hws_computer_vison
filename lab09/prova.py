import cv2
import numpy as np





def prova(verbose=True):
    print('ciao')
    color = np.random.randint(0, 255, (100, 3))
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

        gftt_prev = cv2.goodFeaturesToTrack(cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY), maxCorners=1000,
                                            minDistance=1, useHarrisDetector=True, qualityLevel=0.01)
        # gftt_next = cv2.goodFeaturesToTrack(cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY), maxCorners=1000,
        #                                    minDistance=1, useHarrisDetector=True, qualityLevel=0.01)

        p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY),
                                        gftt_prev, nextPts= None, winSize=(21, 21),
                                        maxLevel= 3, flags=0, minEigThreshold=1e-4)

        mask = np.zeros((fr2.shape[0], fr2.shape[1], fr2.shape[2]), dtype=np.uint8)
        # np.zeros_like(fr2) fa la stessa cosa
        good_p1 = p1[st == 1]
        good_p2 = gftt_prev[st == 1]
        for i, (new, old) in enumerate(zip(good_p1, good_p2)):
            a, b = new.ravel()
            c, d = old.ravel()
            if a > fr2.shape[1] or c > fr2.shape[1]:
                continue
            if b > fr2.shape[0] or d > fr2.shape[0]:
                continue
            cc = color[i % 100].tolist()
            if ((a**2-c**2)+(b**2-d**2))**0.5 > 10.:
                mask = cv2.line(mask, (a, b), (c, d), cc, 4)
                fr2 = cv2.circle(fr2, (a, b), 5, cc, -1)
                fr2 = cv2.circle(fr2, (c, d), 5, cc, -1)
        img = cv2.add(fr2, mask)

        cv2.imshow('frame_final', img)
    if verbose:
        cv2.waitKey(0)


if __name__ == '__main__':

    prova()