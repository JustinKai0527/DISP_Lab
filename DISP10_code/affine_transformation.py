import numpy as np
import cv2 as cv

import numpy as np

def bilinear_interpolation(img, M, N):
    col, row = M.shape
    y, x = img.shape
    m0 = np.floor(M).astype(np.int32)
    n0 = np.floor(N).astype(np.int32)
    
    a, b = M - m0, N - n0
    
    new_img = np.zeros((col, row))
    
    condition = (m0 < 0) | (m0 >= y-1) | (n0 < 0) | (n0 >= x-1)
    
    new_img[condition] = 0
    new_img[~condition] = (1 - a[~condition]) * (1 - b[~condition]) * img[m0[~condition], n0[~condition]] + a[~condition] * (1 - b[~condition]) * img[m0[~condition] + 1, n0[~condition]] + (1 - a[~condition]) * b[~condition] * img[m0[~condition], n0[~condition] + 1] + a[~condition] * b[~condition] * img[m0[~condition] + 1, n0[~condition] + 1]
    
    
    return new_img


if __name__ == "__main__":
    
    img = cv.imread('tower.BMP', cv.IMREAD_GRAYSCALE)
    col, row = img.shape

    M, N = np.arange(0, col), np.arange(0, row)

    N, M = np.meshgrid(N, M)

    T = np.array([[1,0],
                [0.3,1]], dtype=np.float32)

    Cm, Cn = col // 2 , row // 2

    T_inv = np.linalg.inv(T)

    M -= Cm
    N -= Cn

    M, N = T_inv[0, 0] * M + T_inv[0, 1] * N + Cm, T_inv[1, 0] * M + T_inv[1, 1] * N + Cn

    new_img = bilinear_interpolation(img, M, N)
    
    new_img = cv.convertScaleAbs(new_img)
    cv.imshow('image', new_img)
    cv.waitKey()