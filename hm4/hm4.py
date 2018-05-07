from __future__ import print_function
import numpy as np
import cv2


def find_lines_intersection(m_a, c_a, m_b, c_b):
    x = float(c_b - c_a) / float(m_a - m_b)
    y = m_a*x + c_a
    return x, y


def f_lines_intersection(m_a, c_a, x):
    return int(x), int(m_a*x +c_a)


def find_line_param(x_a, y_a, x_b, y_b):
    m = float(y_a - y_b)/float(x_a - x_b)
    c = y_a - m*x_a
    return m, c


def give_y_from_x(m, c, x):
    return int(m*x + c)


def give_x_from_y(m, c, y):
    return int((y/m)-(c/m))


def find_max_for_x(coords, y):
    max_x = 0
    max_m = 0
    max_c = 0
    for coord in coords:
        m, c = find_line_param(coord[0][0], coord[0][1], coord[1][0], coord[1][1])
        if give_x_from_y(m, c, y) > max_x:
            max_x = give_x_from_y(m, c, y)
            max_m = m
            max_c = c
    return max_m, max_c


def find_min_for_x(coords, y):
    min_x = 1000000
    min_m = 0
    min_c = 0
    for coord in coords:
        m, c = find_line_param(coord[0][0], coord[0][1], coord[1][0], coord[1][1])
        if give_x_from_y(m, c, y) < min_x:
            min_x = give_x_from_y(m, c, y)
            min_m = m
            min_c = c
    return min_m, min_c


def find_max_for_y(coords, x):
    max_y = 0
    max_m = 0
    max_c = 0
    for coord in coords:
        m, c = find_line_param(coord[0][0], coord[0][1], coord[1][0], coord[1][1])
        if give_y_from_x(m, c, x) > max_y:
            max_y = give_y_from_x(m, c, x)
            max_m = m
            max_c = c
    return max_m, max_c


def find_max(coords, number):
    max = 0
    for coord in coords:
        if coord[0][number] > max:
            max = coord[0][number]
    return max


def find_min(coords, number):
    min = 1000000
    for coord in coords:
        if coord[0][number] < min:
            min = coord[0][number]
    return min



def find_min_for_y(coords, x):
    min_y = 1000000
    min_m = 0
    min_c = 0
    for coord in coords:
        m, c = find_line_param(coord[0][0], coord[0][1], coord[1][0], coord[1][1])
        if give_y_from_x(m, c, x) < min_y:
            min_y = give_y_from_x(m, c, x)
            min_m = m
            min_c = c
    return min_m, min_c



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


def get_lines_coords(edges, pix_pre, deg_pre, accumulator, min_theta=0, max_theta=0):
    coord = []
    lines = cv2.HoughLines(edges, pix_pre, np.pi / deg_pre, accumulator, min_theta=min_theta, max_theta=max_theta)
    for i in np.arange(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            coord.append([(x1, y1), (x2, y2)])
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


def dief_ing():
    img = cv2.imread('./dief.jpg',)
    cv2.imshow('dief orig ', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    coords_1 = get_lines_coords(edges, 1, 180, 140, np.pi/2, 1.8*np.pi)
    coords_2 = get_lines_coords(edges, 1, 180, 40, 0, np.pi/180)

    min_orizz = find_min_for_y(coords_1, 0)
    max_orizz = find_max_for_y(coords_1, 0)

    x_min = find_min(coords_2, 0)
    x_max = find_max(coords_2, 0)

    print('warp coords')
    lh = f_lines_intersection(min_orizz[0], min_orizz[1], x_min)
    print(lh)

    ll = f_lines_intersection(max_orizz[0], max_orizz[1], x_min)
    print(ll)

    rh = f_lines_intersection(min_orizz[0], min_orizz[1], x_max)
    print(rh)

    rl = f_lines_intersection(max_orizz[0], max_orizz[1], x_max)
    print(rl)

    dst_dim = (380,200)
    h, status = cv2.findHomography(np.array([[lh[0], lh[1]], [ll[0], ll[1]], [rh[0], rh[1]], [rl[0], rl[1]]]),
                                   np.array([[0, 0], [0, dst_dim[1]], [dst_dim[0], 0], [dst_dim[0], dst_dim[1]]]))

    im_dst = cv2.warpPerspective(img, h, dst_dim)
    cv2.imshow('dief warpato ', im_dst)

    ####### PARTE DUE #####

    gray = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150, apertureSize=3)

    coords_1 = get_lines_coords(edges, 1, 90, 110, np.pi / 2-0.1,  np.pi/2+0.1)

    edges = cv2.Canny(gray, 120, 180, apertureSize=3)
    coords_2 = get_lines_coords(edges, 1, 180, 70, 0, np.pi /180)

    edges = cv2.Canny(gray, 50, 120, apertureSize=3)
    coords_3 = get_lines_coords(edges, 2, 180, 70, 0, np.pi / 180)

    x_max = find_max(coords_3, 0)
    x_middle_1 = find_max(coords_2, 0)
    x_middle_2 = find_min(coords_2, 0)
    x_middle_3 = x_middle_2/3
    x_min = 0

    y_max = find_max(coords_1, 1)
    y_min = find_min(coords_1[1:], 1)
    y_middle = (y_max + y_min)/2

    print(x_max, x_middle_1, x_middle_2, x_middle_3, x_min)
    print(y_max, y_middle, y_min)

    print('meccaninca')
    mlh = (x_middle_3, y_min)
    print(mlh)
    mll = (x_middle_3, y_middle)
    print(mll)
    mrh = (x_max, y_min)
    print(mrh)
    mrl = (x_max, y_middle)
    print(mrl)

    print('civile')
    clh = (x_middle_1, y_middle)
    print(clh)
    cll = (x_middle_1, y_max)
    print(cll)
    crh = (x_max, y_middle)
    print(crh)
    crl = (x_max, y_max)
    print(crl)

    print('informatica')
    ilh = (0, y_min)
    print(ilh)
    ill = (0, y_max)
    print(ill)
    imh = (x_middle_3, y_min)
    print(imh)
    iml = (x_middle_3, y_middle)
    print(iml)
    irh = (x_middle_2, y_middle)
    print(irh)
    irl = (x_middle_2, y_max)
    print(irl)

    civ = cv2.imread('./CIV.png', cv2.IMREAD_COLOR)

    h, status = cv2.findHomography(np.array([[0, 0],  [civ.shape[1], 0],[0, civ.shape[0]], [civ.shape[1], civ.shape[0]]]),
                                   np.array([[clh[0], clh[1]+2],  [crh[0], crh[1]+2],[cll[0], cll[1]+2], [crl[0], crl[1]+2]]))

    im_dst_civ = cv2.warpPerspective(civ, h, dst_dim)

    im_dst[im_dst_civ[:, :] != [0, 0, 0]] = im_dst_civ[im_dst_civ[:, :] != [0, 0, 0]]

    mec = cv2.imread('./MEC.png', cv2.IMREAD_COLOR)

    h, status = cv2.findHomography(
        np.array([[0, 0], [0, mec.shape[0]], [mec.shape[1], 0], [mec.shape[1], mec.shape[0]]]),
        np.array([[mlh[0], mlh[1]-2], [mll[0], mll[1]-2], [mrh[0], mrh[1]-2], [mrl[0], mrl[1]-2]]))

    im_dst_mec = cv2.warpPerspective(mec, h, dst_dim)

    im_dst[im_dst_mec[:, :] != [0, 0, 0]] = im_dst_mec[im_dst_mec[:, :] != [0, 0, 0]]

    inf = cv2.imread('./INF.png', cv2.IMREAD_COLOR)

    h, status = cv2.findHomography(
        np.array([[0, 0],
                  [0, inf.shape[0]],
                  [inf.shape[1], 0],
                  [inf.shape[1], int(inf.shape[0]/2)],
                  [inf.shape[1], inf.shape[0]]]),
        np.array([[ilh[0], ilh[1]],
                  [ill[0], ill[1]],
                  [imh[0], imh[1]],
                  [iml[0], iml[1]],
                  [irl[0], irl[1]]]))

    im_dst_inf = cv2.warpPerspective(inf, h, dst_dim)
    cv2.imshow('inf', im_dst_inf)

    im_dst[im_dst_inf[:, :] != [0, 0, 0]] = im_dst_inf[im_dst_inf[:, :] != [0, 0, 0]]



    cv2.imshow('finale', im_dst)

    for coord in coords_3:
        cv2.line(im_dst, coord[0], coord[1], (0, 0, 255), 2)
    cv2.imshow('dief ', im_dst)
    cv2.waitKey(0)


def prova():
    img = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab05/houghf/5.BMP', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    cv2.imshow('e before', gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('canny', edges)

    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(gray, kernel, iterations=1)
    img_dilate = cv2.dilate(gray, kernel, iterations=1)
    tmp = img_erosion -img_dilate
    cv2.imshow('erosion', img_erosion -img_dilate)


    lines = cv2.HoughLines(edges, 2, np.pi / 180,80)

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
    cimg = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab05/houghf/5.BMP', cv2.IMREAD_GRAYSCALE)
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
    dief_ing()
    #a_caso()
    #main('image.jpg', 'ciao')
    #prova()