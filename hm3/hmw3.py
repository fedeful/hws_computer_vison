# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from facilities.others import colored_function
from lab01.linears import fast_negative
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from os.path import join,abspath,isdir,isfile
from os import getcwd,listdir, chdir
from sklearn import metrics
from sklearn.externals import joblib
import cv2
import numpy as np
import PIL.Image
from sklearn import svm
import operator
import argparse
import cv2
import numpy as np


def ghiralndina(imagec, place):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    # image = cv2.imread('/home/federico/PycharmProjects/hws_computer_vison/lab03/img/modena_skyline_02.png',
    # cv2.IMREAD_COLOR)
    image = imagec.copy()
    image = image.astype(np.float64)

    # loop over the number of segments
    s = []
    seg = []
    for numSegments in (80, 90, 100):
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments=numSegments, sigma=5)
        val = segments[(place)]
        a = set()
        tmp = np.where(segments == val)
        for i in np.arange(0, len(tmp[0])):
            a.add((tmp[0][i], tmp[1][i]))
        s.append(a)
    return s


def find_middle(img):
    index = img.shape[0]/2 - 40
    while True:

        if index >= img.shape[0]:
            return -1

        if img[index-1, 0] != img[index, 0]:
            return index

        index += 1


def find_kmeans_middle(img):
    base = img.shape[0] / 10
    index = img.shape[0] / 2 - base
    while True:
        if index >= img.shape[0]:
            return -1
        if img[index - 1, 0] != img[index, 0]:
            return index
        index += 1


def find_laplacian_middle_form_kmeans(img, start, bsize, th):

    index = start
    while True:
        if index >= img.shape[0]:
            break
        if img[index] >= th:
            break
        index += 1
    return 0


def find_box(img, line_search, box_size):
    colo = img[line_search, 0]

    for i in np.arange(int(box_size[0] / 2.), img.shape[1]-int(box_size[0] / 2.)):
        counter = 0
        white = 0
        black = 0
        for k in np.arange(-int(box_size[0] / 2.), box_size[0]/2):
            for l in np.arange(-int(box_size[1] / 2), box_size[1] / 2):
                if img[line_search + k, i + l] == 0:
                    black += 1
                else:
                    white += 1
        if white >= (box_size[0]*box_size[1])*(8./10):
            return i, 255
        elif black >= (box_size[0]*box_size[1])*(8./10):
            return  i,0
    return -1


def kmeans(data,dim=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.array(data, dtype=np.float32), dim, None, criteria, 50, flags)
    return labels


def apply_kmeas(img,dim=2, x_flag=True, y_flag=False):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    z = np.reshape(gray, (-1, 1))
    z = np.append(z, np.reshape(hsv[:, :, 1], (-1, 1)), axis=1)
    z = np.append(z, np.reshape(hsv[:, :, 0], (-1, 1)), axis=1)
    z = np.append(z, np.reshape(hsv[:, :, 2], (-1, 1)), axis=1)
    z = np.append(z, np.reshape(img[:, :, 0], (-1, 1)), axis=1)
    z = np.append(z, np.reshape(img[:, :, 1], (-1, 1)), axis=1)
    z = np.append(z, np.reshape(img[:, :, 1], (-1, 1)), axis=1)

    if x_flag:
        t = [i for i in np.arange(0, img.shape[0])]
        tmp = np.array(t)
        tmp = np.reshape(tmp, (-1, 1))
        x = np.zeros((img.shape[0], img.shape[1]))
        x += tmp
        z = np.append(z, np.reshape(x, (-1, 1)), axis=1)

    lab = kmeans(z)
    lab = np.reshape(lab, (img.shape[0], img.shape[1]))

    if y_flag:
        t = [i for i in np.arange(0, img.shape[1])]
        tmp = np.array(t)
        #tmp = np.reshape(tmp, (-1, 1))
        x = np.zeros((img.shape[0], img.shape[1]))
        x += tmp
        z = np.append(z, np.reshape(x, (-1, 1)), axis=1)

    lab = kmeans(z,dim)
    lab = np.reshape(lab, (img.shape[0], img.shape[1]))

    lab[lab == 1] = 255
    lab[lab == 2] = 128

    return lab.astype(np.uint8)


def apply_detec(img_gray):
    img2 = img_gray.copy()
    template = cv2.imread('/home/federico/Desktop/ghird.png', 0)
    w, h = template.shape[::-1]
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    dd ={}


    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        if top_left in dd:
            dd[top_left] += 1
        else:
            dd[top_left] = 0
        bottom_right = (top_left[0] + w, top_left[1] + h)

        '''
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()'''
    tl = max(dd.iteritems(), key=operator.itemgetter(1))[0]
    return tl, (tl[0] + w, tl[1] + h)


def day_night_row(path, label):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    histb = cv2.calcHist([img[:, :, 0]], [0], None, [256], [0, 256])
    histg = cv2.calcHist([img[:, :, 1]], [0], None, [256], [0, 256])
    histr = cv2.calcHist([img[:, :, 2]], [0], None, [256], [0, 256])

    histb = np.resize(histb, (1, -1))
    histg = np.resize(histg, (1, -1))
    histr = np.resize(histr, (1, -1))
    row = np.concatenate((histb, histg, histr), axis=1)

    row = row[0].tolist()
    if label >= 0:
        row.append(label)
    row = np.array([row])

    return row


def findfiles(path, range_day, range_night):
    files_list = []
    prev_path = getcwd()
    chdir(path)
    day = 1
    night = 0
    # print(prev_path)

    fname = 'modena_skyline_{}.png'.format(range_day[0])
    matrix = day_night_row(join(path, fname), day)

    for i in np.arange(range_day[0]+1, range_day[1]):
        fname = 'modena_skyline_{}.png'.format(i)
        tmp = day_night_row(join(path, fname), day)
        matrix = np.concatenate((matrix, tmp), axis=0)

    for i in np.arange(range_night[0]+1, range_night[1]):
        fname = 'modena_skyline_{}.png'.format(i)
        tmp = day_night_row(join(path, fname), night)
        matrix = np.concatenate((matrix, tmp), axis=0)

    chdir(prev_path)

    return matrix


def class_day_night(path, train_mode=False):

    if train_mode:
        train_dataset = findfiles(path, (52, 111), (112, 172))
        test_dataset = findfiles(path, (187, 233), (245, 317))

        y = train_dataset[:, -1]
        X = train_dataset[:, 0:-1]
        yt = test_dataset[:, -1]
        Xt = test_dataset[:, 0:-1]

        clf = svm.SVC(kernel='linear', C=1, gamma=0.01)
        clf.fit(X, y)
        yp = clf.predict(Xt)
        print(metrics.accuracy_score(yp, yt))
        #94%

        joblib.dump(clf, './daynightclass.pkl')

        return 3
    else:
        clf = joblib.load('./daynightclass.pkl')
        u = day_night_row(path, -1)
        yp = clf.predict(u)
        if yp[0] == 1:
            return 'day'
        else:
            return 'night'


def main(filename, PRINT_MODE=True):

    img_colored = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_colored, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

    if PRINT_MODE:
        cv2.imshow('original color image ', img_colored)

    img_proc = cv2.morphologyEx(img_colored, cv2.MORPH_CLOSE, np.ones((3, 3)))
    img_proc = cv2.bilateralFilter(img_proc, 3, 100, 100)

    if PRINT_MODE:
        cv2.imshow('morph close and bilateral filter image', img_proc)

    first_kmeans = apply_kmeas(img_proc)

    if PRINT_MODE:
        cv2.imshow('first kmeans', first_kmeans)

    mid = find_middle(first_kmeans)
    box = find_box(first_kmeans, mid, (40, 2))

    print(mid, box)

    u = img_gray[mid - 190:mid + 150, :]
    a, b = apply_detec(u)
    if PRINT_MODE:
        cv2.imshow('ghir trovata', img_colored[(mid - 190) + a[1]:(mid - 190) + b[1], a[0]:b[0]])
    ghirt = img_colored[(mid - 190) + a[1]:(mid - 190) + b[1], a[0]:b[0]]
    ghirt = cv2.morphologyEx(ghirt, cv2.MORPH_CLOSE, np.ones((4, 4)))
    ghirt = cv2.bilateralFilter(ghirt, 4, 100, 100)
    edges = cv2.Canny(ghirt, 180, 200)
    if PRINT_MODE:
        cv2.imshow('ghir trovatabil', edges)
    uu = apply_kmeas(ghirt, 2, False, True)

    if PRINT_MODE:
        cv2.imshow('kmeans on found image', uu)

    a2 = first_kmeans.copy()

    realbord = img_colored[mid - 100: mid + 100, :, :]
    # rb = cv2.morphologyEx(realbord, cv2.MORPH_CLOSE, np.ones((3, 3)))
    realbord = cv2.bilateralFilter(realbord, 3, 100, 100)
    # if PRINT_MODE:
    #    cv2.imshow('box image', a)
    uu = apply_kmeas(realbord, 2, True)
    # a,uu = cv2.threshold(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

    # cv2.imshow('kmenas2', uu)
    first_kmeans[mid - 100: mid + 100, :] = uu
    cv2.imshow('final2', first_kmeans)

    k = cv2.waitKey(0)
    if k == 'q':
        cv2.destroyAllWindows()


if __name__ == '__main__':

    print(class_day_night('/home/federico/Desktop/img/modena_skyline_370.png'))


