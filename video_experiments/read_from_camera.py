from __future__ import print_function
from lab01.histograms import otsu
from lab01.linears import apply_threshold,apply_mask_on_colored_image
import numpy as np
import cv2
import time



#actuallly it takes too much time
#I'm going to add an equal function that uses only opencv lib
def read_video_from_file():
    import numpy as np
    import cv2

    cap = cv2.VideoCapture('20180318_110247.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640, 480))
        th = otsu(gray)
        dire, inv = apply_threshold(gray, th)
        w, b = apply_mask_on_colored_image(cv2.resize(frame, (640, 480)), inv)
        #cv2.imshow('frame', w)
        out.write(w)
        #time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def read_video_from_camera():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_video_from_file()