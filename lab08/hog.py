import cv2
import numpy as np

Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)*(1/8.)
Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)*(1/8.)
lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def gradients(img, kernel):
    grad = cv2.filter2D(img, -1, kernel)
    return grad


def magnitude(Gx, Gy):
    tmp = (Gx**2 + Gy**2)**0.5
    norm = ((255.*3/8)**2+(255.*3/8)**2)**0.5

    return (tmp/norm)*255


def theta(Gx, Gy):
    return np.arctan2(Gy, Gx)


def histogram(img, nbin=256, norm=True):
    h = np.zeros(nbin)

    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if img[i, j, 0] == np.pi:
                h[0] += img[i, j, 1]
            else:
                h[int((img[i, j, 0] / (2*float(np.pi))) * nbin)] += img[i, j, 1]

    return h


def sobel(file_name):

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = img.astype(dtype=np.float64)

    gradx = gradients(img, Sx)
    grady = gradients(img, Sy)

    H = theta(gradx, grady)
    V = magnitude(gradx, grady)
    S = np.ones(V.shape, dtype=np.float64) * 255


    #H = (180. - 0) / (np.amax(H) - np.amin(H)) * (H - np.amin(H))
    H = H + np.pi
    im = np.zeros((H.shape[0], H.shape[1], 3), dtype=np.float64)
    im[:, :, 0] = H
    im[:, :, 1] = S
    im[:, :, 2] = V

    hsv = im
    hsv = hsv.astype(dtype=np.uint8)
    #hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow('magnitude', hsv)
    #cv2.waitKey(0)
    return hsv


def calcHOG(src, wCell, hCell, nBinAngle):
    lh = []

    wsize = src.shape[0]/hCell
    hsize = src.shape[1]/wCell

    for row in np.arange(0, hCell*(hsize-1), hCell):
        for col in np.arange(0, wCell*(wsize-1), wCell):
            tmp = histogram(src[row:row+hCell, col:col+wCell], nBinAngle, True)
            if len(lh) == 0:
                lh = tmp
            else:
                lh = np.append(lh, tmp, axis=0)

    return lh


if __name__ == '__main__':
    hsv = sobel('man.jpg')
    l = calcHOG(hsv, 8, 8, 10)
    print('ciao')
