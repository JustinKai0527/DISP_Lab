import numpy as np
import cv2 as cv

def bilinear_interpolation(img_file):
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    cv.imshow('image', img)
    cv.waitKey()
    # doing the bilinear interpolation
    y_factor = 1.6
    x_factor = 1.5

    col, row = img.shape

    # notice that is col, row - 1 this can prevent the index out of range when facing edge
    new_col, new_row = int(col * y_factor), int(row * x_factor)
    m = np.arange(0, new_col)     # m = y
    n = np.arange(0, new_row)     # n = x

    n, m = np.meshgrid(n, m)
    m = m.astype(np.float32)
    n = n.astype(np.float32)

    m /= y_factor
    n /= x_factor
    # print(m[0,:], n[:,0])

    m0 = np.floor(m).astype(np.int32)
    n0 = np.floor(n).astype(np.int32)

    a, b = m - m0, n - n0

    new_img = np.zeros((new_col, new_row))
    img = np.pad(img, ((0, 1), (0, 1)), mode='edge')
    # doing the bilinear interpolation after padding the edge 
    new_img = (1 - a) * (1 - b) * img[m0, n0] + a * (1 - b) * img[m0 + 1, n0] + (1 - a) * b * img[m0, n0 + 1] + a * b * img[m0 + 1, n0 + 1]

    new_img = cv.convertScaleAbs(new_img)
    cv.imshow('image', new_img)
    cv.waitKey()
    
    return    

if __name__ == "__main__":
    
    bilinear_interpolation("BABOON.BMP")

