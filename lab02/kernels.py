from __future__ import print_function
from facilities.others import colored_function
import numpy as np
import cv2


def generate_kernel(dim, value):
    kernel = np.ones((dim, dim), dtype=np.float64)/value
    return kernel


def gaussian_kernel2d(ks, variance=1):
    print(ks)
    gauss = np.zeros((ks, ks), dtype=np.float64)

    for i in np.arange(0, ks):
        for j in np.arange(0, ks):
            num_esp = np.float64(((i-(ks/2))**2)+((j-ks/2)**2))
            den_esp = np.float64((variance**2)*2.0)
            den = np.float64(1)/np.float64(2.0*np.pi*(variance**2.0))
            gauss[i, j] = np.exp(-num_esp/den_esp)*den

    return gauss/np.sum(gauss)


def gaussian_kernel1d(ks, variance=1):
    gauss = np.zeros(ks, dtype=np.float64)

    ic = -int(ks/2)
    for i in np.arange(0, ks):
        num_esp = np.float64((ic**2))
        den_esp = np.float64((variance**2)*2.0)
        den = np.float64(1)/np.float64(np.sqrt(2.0*np.pi)*np.float64(variance))
        gauss[i] = np.exp(-num_esp/den_esp)*den
        ic += 1

    return gauss/np.sum(gauss)


def fast_gaussian_kernel2d(ks, variance=1):

    x, y = np.mgrid[-(ks/2):((ks/2)+1), -(ks/2):((ks/2)+1)]
    num_esp = np.float64((x**2) + (y**2))
    den_esp = np.float64((variance ** 2) * 2.0)
    den = np.float64(2.0 * np.pi * (variance ** 2.0))
    gauss = np.exp(-num_esp / den_esp) * (1.0 / den)

    return gauss/np.sum(gauss)


def convolution(img, kernel):
    conv = np.zeros(img.shape, dtype=np.float64)
    for i in np.arange(kernel.shape[0]/2, img.shape[0]-kernel.shape[0]/2):
        for j in np.arange(kernel.shape[1]/2, img.shape[1]-kernel.shape[1]/2):
            for k in np.arange(-int(kernel.shape[0]/2), kernel.shape[0]/2+1):
                for l in np.arange(-int(kernel.shape[1]/2), kernel.shape[1]/2+1):
                    conv[i, j] += img[i+k, j+l]*kernel[k+kernel.shape[1]/2, l+kernel.shape[1]/2]

    return conv.astype(np.uint8)


def median(img, ker_dim=(3, 3)):
    med = np.zeros(img.shape, dtype=np.float64)
    for i in np.arange(ker_dim[0]/2, img.shape[0]-ker_dim[0]/2):
        for j in np.arange(ker_dim[1]/2, img.shape[1]-ker_dim[1]/2):
            tmp = img[i-int(ker_dim[0]/2):i+(ker_dim[0]/2)+1, j-int(ker_dim[1]/2):j+(ker_dim[1]/2)+1]
            tmp = np.resize(tmp, (-1, 1))
            med[i, j] = np.median(tmp)

    return med.astype(np.uint8)


def bilateral(img, ks, sigmad, sigmar):
    bil = np.zeros(img.shape, dtype=np.float64)

    for i in np.arange(ks/2, img.shape[0]-int(ks/2)):
        for j in np.arange(ks/2, img.shape[1]-int(ks/2)):
            num = np.float64(0)
            den = np.float64(0)
            for k in np.arange(-int(ks/2), ks/2+1):
                for l in np.arange(-int(ks/2), ks/2+1):
                    dk = -(((np.float64(i)-np.float64(i+k))**2 + (np.float64(j)-np.float64(j+l))**2) /
                           np.float64(2*(sigmad**2)))
                    rk = -(((np.float(img[i, j])-np.float64(img[i+k, j+l]))**2)/(2*np.float64(sigmar**2)))
                    w = np.float64(np.exp(dk + rk))
                    num += np.float64(img[i+k, j+l])*w
                    den += w

            bil[i, j] = num/den

    return bil.astype(np.uint8)


def fourier(mat):
    four = np.zeros(mat.shape, dtype= np.complex)
    print(mat.shape)
    for k in np.arange(-(mat.shape[0]/2),(mat.shape[0]/2)+1):
        for l in np.arange(-(mat.shape[1]/2),(mat.shape[1]/2)+1):
            tmp = 0.0
            for i in np.arange(-(mat.shape[0]/2),(mat.shape[0]/2)+1):
                for j in np.arange(-(mat.shape[1]/2),(mat.shape[1]/2)+1):
                    print(k,l,i,j)
                    e1 = float(k*i)/float(mat.shape[0])
                    e2 = float(l*j)/float(mat.shape[1])
                    tmp += mat[i+mat.shape[0]/2,j+mat.shape[1]/2]*np.exp(-(1j)*2*np.pi*(e1+e2))

            four[k+mat.shape[0]/2,l+mat.shape[1]/2] = tmp
    return four

def fourier2(mat):
    four=np.zeros(mat.shape, dtype= np.complex)
    print(mat.shape)
    for k in np.arange(0,mat.shape[0]):
        for l in np.arange(0,mat.shape[1]):
            tmp = np.complex(0.0)
            for i in np.arange(0,mat.shape[0]):
                for j in np.arange(0,mat.shape[1]):
                    #print(k,l,i,j)
                    e1 = np.complex(k*i)/np.complex(mat.shape[0])
                    e2 = np.complex(l*j)/np.complex(mat.shape[1])
                    tmp += np.complex(mat[i, j])*np.exp(-1j*np.complex(2)*np.complex(np.pi)*np.complex(e1+e2))

            four[k,l] = tmp
    return four

def fouriershift(mat):
    fours = np.zeros(mat.shape, dtype=np.complex)
    N = (mat.shape[0])
    for i in np.arange(0, mat.shape[0]):
        for j in np.arange(0, mat.shape[1]):
            fours[(i+N/2)%N,(j+N/2)%N] = mat[i,j]
    return fours


