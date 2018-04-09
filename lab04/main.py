from __future__ import print_function
import cv2
import numpy as np

label_idx = 1
idx = 1


def is_foreground(pixel):
    if pixel != 0:
        return True
    return False


def not_labeled(pixel):
    if pixel >0 and pixel != 255:
        return False
    return True


def ffilter(img):
    image = img.copy()
    label_idx = 1
    queue = []
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if is_foreground(img[i, j]) and not_labeled(img[i, j]):
                image[i, j] = label_idx
                queue.append((i, j))
                image = queue_solver(image, queue, label_idx)
                label_idx += 1
    return image


def queue_solver(img, queue, lab):

    while True:
        pix = queue[0]
        for qsi in np.arange(-1, 2):
            for qsj in np.arange(-1, 2):
                if qsi != 0 or qsj != 0:
                    if pix[0] + qsi >= 0 and pix[0] + qsi < img.shape[0] and pix[1] + qsj >= 0 and pix[1] + qsj < img.shape[1]:
                        if is_foreground(img[pix[0]+qsi, pix[1]+qsj]) and not_labeled(img[pix[0] + qsi, pix[1]+qsj]):
                            img[pix[0]+qsi, pix[1]+qsj] = lab
                            queue.append((pix[0]+qsi, pix[1]+qsj))

        queue.pop(0)
        if len(queue) == 0:
            break
    return img


if __name__ == '__main__':
    img = cv2.imread('./binary.png', cv2.IMREAD_GRAYSCALE)
    img[img[:,:] != 0] = 255

    label_idx = 1
    queue = []
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if is_foreground(img[i, j]) and not_labeled(img[i, j]):
                img[i, j] = label_idx
                queue.append((i, j))
                while True:
                    pix = queue[0]
                    for qsi in np.arange(-1, 2):
                        for qsj in np.arange(-1, 2):
                            if qsi != 0 or qsj != 0:
                                if pix[0] + qsi >= 0 and pix[0] + qsi < img.shape[0] and pix[1] + qsj >= 0 and pix[1] + qsj < img.shape[1]:
                                    if is_foreground(img[pix[0] + qsi, pix[1] + qsj]) and not_labeled(img[pix[0] + qsi, pix[1] + qsj]):
                                        img[pix[0] + qsi, pix[1] + qsj] = label_idx
                                        queue.append((pix[0] + qsi, pix[1] + qsj))

                    queue.pop(0)
                    if len(queue) == 0:
                        break
                label_idx += 1


    #img = ffilter(img)
    #np.zeros((img.shape[0],))
    ff = np.zeros((img.shape[0],img.shape[1], 3))
    ff[img[:, :] == 1] = [255, 0, 0]
    ff[img[:, :] == 2] = [180, 0, 0]
    ff[img[:, :] == 3] = [130, 0, 0]
    ff[img[:, :] == 4] = [70, 0, 0]
    ff[img[:, :] == 5] = [0, 255, 0]
    ff[img[:, :] == 6] = [0, 180, 0]
    ff[img[:, :] == 7] = [0, 130, 0]
    ff[img[:, :] == 8] = [0, 70, 0]
    ff[img[:, :] == 9] = [0, 0, 255]
    ff[img[:, :] == 10] = [0, 0, 180]
    ff[img[:, :] == 11] = [0, 0, 130]
    ff[img[:, :] == 12] = [0, 0, 70]
    ff[img[:, :] == 13] = [255, 0, 0]
    ff[img[:, :] == 14] = [100, 100, 150]
    ff[img[:, :] == 15] = [150, 0, 0]
    ff[img[:, :] == 16] = [255, 255, 255]
    #a = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    cv2.imshow('prova', ff)
    cv2.waitKey(0)






