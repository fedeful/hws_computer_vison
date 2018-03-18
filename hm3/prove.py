from __future__ import print_function
import cv2
import numpy as np
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':
    a = cv2.imread('/home/fede/PycharmProjects/computer_vision/hm3/modena.jpg', cv2.IMREAD_COLOR)
    #a = cv2.resize(a,(128,128))
    #cv2.imshow('prova1', a)
    #cv2.waitKey(0)

    a = cv2.bilateralFilter(a, 11, 75, 75)



    '''
    lab = np.zeros((a.shape[0]*a.shape[1], 1))
    u = []
    for i in np.arange(0,a.shape[0]):
        for j in np.arange(0, a.shape[1]):
            tmp=[]
            tmp.append(i)
            tmp.append(j)
            for k in np.arange(0, 3):
                tmp.append(a[i, j, k])
            u.append(tmp)


    


    for i in np.arange(0, a.shape[0]*a.shape[1]):
        if bestlab[i] == 0:
            a[u[i][0],u[i][1]] = [255,0,0]
        elif bestlab[i] == 1:
            a[u[i][0],u[i][1]] = [0,255,0]
        else:
            a[u[i][0], u[i][1]] = [0, 0, 255]'''
    cv2.imshow('prova',a)
    cv2.waitKey(0)
    print('ciao')

