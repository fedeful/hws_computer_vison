import cv2
import numpy as np

def main2(fname):
    cap = cv2.VideoCapture(0)
    #ret, frame = cap.read()
    id_f = 0
    l_data = []
    w_data = []
    if cap.isOpened() == True:
        while cap.isOpened():

            ret, frame = cap.read()
            if ret == True:
                if id_f % 10 == 0:
                    f2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier('./pesi.xml')
                    faces = face_cascade.detectMultiScale(f2)
                    l_data = []
                    w_data = []
                    for (x, y, w, h) in faces:

                        l_data.append(frame[y:y+h, x:x+w])
                        w_data.append((w, h))
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
                else:
                    for i in np.arange(0,len(l_data)):
                        res = cv2.matchTemplate(frame, l_data[i], cv2.TM_SQDIFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        frame = cv2.rectangle(frame, (min_loc[0], min_loc[1]), (min_loc[0] + w_data[i][0] , min_loc[1] +
                                                                                w_data[i][1]), color=(0, 0, 255), thickness=5)
            id_f = id_f +1
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()




def main(fname):

    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(fname, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier('./pesi.xml')
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        color = cv2.rectangle(color, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=5)

    cv2.imshow('nome', color)
    cv2.waitKey(0)


if __name__ == '__main__':
    name = 'b.jpg'
    name2 = 'Buffy.mp4'
    main(name)