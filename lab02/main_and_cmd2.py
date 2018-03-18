
'''
def caso:
    a = gaussianf(21, 0, 5)
    # b = opgaussian(21,0,1)
    b = fourier2(a)
    # plt.imshow(a,cmap='gray')
    # plt.colorbar()

    f = np.fft.fft2(a)
    print(b)
    print(' \n\n')
    print(f)
    print(np.array_equiv(f, b))
    fshift = np.fft.fftshift(f)
    for i in np.arange(0, f.shape[0]):
        for j in np.arange(0, f.shape[1]):
            print(b[i, j])
            print(f[i, j])
            print(b[i, j] == f[i, j])
    fs = fouriershift(b)
    ms1 = 20 * np.log(np.abs(fs))
    ms2 = 20 * np.log(np.abs(fshift))
    plt.subplot(131)
    plt.imshow(a)
    plt.colorbar()
    plt.title('gaussian')

    plt.subplot(132)
    plt.imshow(ms1)
    plt.title('ms1')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(ms2)
    plt.title('ms2')
    # plt.imshow(b)
    plt.colorbar()
    plt.show()
    # print(np.sum(a))
    # print(a)
    # print(b)



if __name__ == '__main__':
    ia = cv2.imread('/home/fede/PycharmProjects/computer_vision/lab02/img/volto_rughe.jpg', cv2.IMREAD_COLOR)
    fi = colored_function(bilateral, ia, 5, 80, 80)
    cv2.imshow('meno rughe', fi)
    qui = cv2.waitKey(0)
    if qui == 'q':
        cv2.destroyAllWindows()
'''