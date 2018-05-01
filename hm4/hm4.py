from __future__ import print_function
import numpy as np
import cv2


def find_lines_intersection(m_a, c_a, m_b, c_b):
    x = float(c_b - c_a) / float(m_a - m_b)
    y = m_a*x + c_a
    return x, y


def f_lines_intersection(m_a, c_a, x):
    return int(m_a*x +c_a)


def find_line_param(x_a, y_a, x_b, y_b):
    m = float(y_a - y_b)/float(x_a - x_b)
    c = y_a - m*x_a
    return m, c


def give_y_from_x(m, c, x):
    return int(m*x + c)


def give_x_from_y(m, c, y):
    return int((y/m)-(c/m))


def get_lines_params(edges, theta, rho):
    coord = []
    lines = cv2.HoughLines(edges, 1, np.pi / theta, rho)
    for i in np.arange(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            coord.append((x1, y1, x2, y2))
    return coord


def main(in_file, out_file, smode=True):
    print('ciao')

    img = cv2.imread(in_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if smode:
        cv2.imshow('edges image ', edges)

    #retta alta, 160gradi
    r160h = get_lines_params(edges, 160, 130)
    min_y = 600
    m0 = 0
    c0 = 0
    for coord in r160h:
        m, c = find_line_param(coord[0], coord[1], coord[2], coord[3])
        yy = give_y_from_x(m, c, img.shape[1])
        if yy < min_y:
            min_y = yy
            m0 = m
            c0 = c

    #retta bassa, 160

    r160h = get_lines_params(edges, 90, 180)
    max_y = 0
    m1 = 0
    c2 = 0
    for coord in r160h:
        m, c = find_line_param(coord[0], coord[1], coord[2], coord[3])
        yy = give_y_from_x(m, c, img.shape[1])
        if yy > max_y:
            max_y = yy
            m1 = m
            c1 = c


    #retta vert, dx
    r160h = get_lines_params(edges, 1, 50)
    max_x = 0
    for coord in r160h:
        if coord[0] > max_x:
            max_x = coord[0]
    print(f_lines_intersection(m0,c0,max_x))
    print(f_lines_intersection(m1, c1, max_x))


    # retta vert, sx
    r160h = get_lines_params(edges, 1, 50)
    min_x = 0
    for coord in r160h:
        if (max_x - 50)>coord[0] > min_x:
            min_x = coord[0]
    print(f_lines_intersection(m0, c0, min_x))
    print(f_lines_intersection(m1, c1, min_x))
    print(min_y, max_y, max_x, min_x)

    book_img = cv2.imread('./higuagoal.jpg', cv2.IMREAD_COLOR)
    book_img = cv2.resize(book_img, (200, 300))

    h, status = cv2.findHomography(np.array([[0, 0], [0, 300], [200, 0], [200, 300]]),
                                   np.array([[min_x, 74], [min_x, 205], [max_x, 38], [max_x, 212]]))

    im_dst = cv2.warpPerspective(book_img, h, (img.shape[1], img.shape[0]))

    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if not np.array_equal(im_dst[i,j], [0, 0, 0]):
                img[i, j] = im_dst[i, j]
    cv2.imshow('higua', img)


    if smode:
        cv2.waitKey(0)

def a_caso():
    book_img = cv2.imread('./kali.png', cv2.IMREAD_COLOR)
    book_img = cv2.resize(book_img, (200, 300))

    h, status = cv2.findHomography(np.array([[0, 0], [0, 300], [200, 0], [200, 300]]),
                                   np.array([[50, 50], [100, 100], [100, 50], [150, 100]]))

    # h, status = cv2.findHomography(np.array([[0, 0],[0,300],[200,0],[200,300]]),np.array([[50,50],[100,100],[100,50],[150,100]]))

    ''' 
    The calculated homography can be used to warp 
    the source image to destination. Size is the 
    size (width,height) of im_dst
    '''

    # im_dst = cv2.warpPerspective(book_img, h, (300,250))
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('piccadilly ', edges)
    # retta bassa
    #lines = cv2.HoughLines(edges,1,np.pi/160, 220)
    #lines = cv2.HoughLines(edges, 1, np.pi /180, 50,min_theta=0,max_theta=np.pi/180)
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 150, min_theta=np.pi/5, max_theta=3*np.pi / 4)
    # lines = cv2.HoughLines(edges,1,np.pi/90, 180)
    # rette verticali
    # lines = cv2.HoughLines(edges,1,np.pi, 50)
    #lines = cv2.HoughLines(edges, 1, (np.pi / 180) * 40, 100)

    max_x = 515
    tmp_max = 0
    max_y = 600
    #lines = cv2.HoughLines(edges, 1, np.pi, 50)
    for i in np.arange(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))


            # if (max_x - 50) > x1 > tmp_max:
            #    tmp_max = x1
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(max_y)
    cv2.imshow('libro ', img)
    cv2.waitKey(0)


def prova():
    img = cv2.imread('/home/federico/PycharmProjects/hws_computer_vison/lab05/houghf/3.BMP', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 2, np.pi / 180,90)

    for i in np.arange(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))


            # if (max_x - 50) > x1 > tmp_max:
            #    tmp_max = x1
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cimg = cv2.imread('/home/federico/PycharmProjects/hws_computer_vison/lab05/houghf/3.BMP', cv2.IMREAD_GRAYSCALE)
    ret2, th2 = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('otsu',th2)
    circles = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=10, minRadius=10, maxRadius=40)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('cerchi', cimg)
    cv2.imshow('libro ', img)

    cv2.waitKey(0)


if __name__ == '__main__':
    #a_caso()
    #main('image.jpg', 'ciao')
    prova()