from __future__ import print_function
from os.path import join,abspath,isdir,isfile
from os import getcwd,listdir, chdir
import cv2
import numpy as np
import PIL.Image


def files(list_files):
    f = []
    for i, j in list_files:
        d = {}
        if i.endswith('img.png'):
            d['i'] = cv2.imread(i , cv2.IMREAD_COLOR)
            d['l'] = np.asarray(PIL.Image.open(j))

        else:
            d['i'] = cv2.imread(j, cv2.IMREAD_COLOR)
            d['l'] = np.asarray(PIL.Image.open(i))
        f.append(d)

    return f


def files_path(path):
    files_list =[]
    prev_path = getcwd()

    chdir(path)
    #print(prev_path)

    for f in listdir(path):
        if isdir(abspath(f)):
            pre = getcwd()
            chdir(abspath(f))
            duo =[]
            for elem in listdir(getcwd()):
                if elem.startswith('img.'):
                    #print(abspath(elem))

                    duo.append(abspath(elem))
                if elem.startswith('label.'):
                    #print(abspath(elem))
                    duo.append(abspath(elem))
            chdir(pre)
            if(len(duo)==2):
                files_list.append(duo)

    chdir(prev_path)
    return files_list


def files_(path):
    files_list =[]
    prev_path = getcwd()

    chdir(path)
    #print(prev_path)

    for f in listdir(path):
        if isfile(abspath(f)):
            elem = abspath(f)
            arr = np.asarray(PIL.Image.open(elem))
            cv2.imwrite(elem, arr.astype(np.uint8))
    chdir(prev_path)
    return files_list




def read_from_file():
    import numpy as np
    import cv2

    cap = cv2.VideoCapture('/home/federico/Desktop/img/Modena timelapse gennaio 2017-0gfOTtUK0zQ.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    val = 10
    while (cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            break

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = cv2.resize(frame, (640, 480))

        cv2.imwrite('/home/federico/Desktop/img/modena_skyline_%d.png'%val, w)
        val +=1

        # time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()




if __name__ == '__main__':
    a = files_('/home/federico/Desktop/Label')