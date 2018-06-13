import cv2
from hm5.detection import Imb
cap = cv2.VideoCapture('http://admin:123456@neuralstory-host.ing.unimore.it/camera/videostream.cgi?rate=1')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


imb = Imb()


if cap.isOpened():
    while True:
        ret, frame = cap.read()

        tmp = imb.get_image_with_box(frame)

        out.write(tmp)
        cv2.imshow('Video', tmp)

        if cv2.waitKey(1) == 27:
            exit(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
