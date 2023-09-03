import numpy as np
import cv2 as cv


img = cv.imread('peppers.bmp', cv.IMREAD_GRAYSCALE)
img2 = img
h, w = img.shape
x, y = h//2, w//2

rx = 100
ry = 150
h = np.arange(0, h)
w = np.arange(0, w)
row, col = np.meshgrid(w, h)

matrix = (((row - x) / rx) ** 2 + ((col - y) / ry) ** 2) > 1

img[matrix] = 0

cv.imshow('image', img)
cv.waitKey()

# norm

print("L0 norm", np.sum(img2 != 0))
print("L1 norm", np.sum(np.abs(img2)))
print("L2 norm", np.sqrt(np.sum(np.square(img2))))

# central moment

def central_moment(img, a=1, b=1):
    
    print(a, b, sep=' ', end=None)
    N, M = img.shape
    h = np.repeat(np.linspace(0, N-1, N).reshape(-1, 1), M, axis=1)
    w = np.repeat(np.linspace(0, M-1, M).reshape(1, -1), N, axis=0)

    nx_bar = np.sum(w * img) / np.sum(img)
    ny_bar = np.sum(h * img) / np.sum(img)


    m = np.sum(np.power(w - nx_bar, a) * np.power(h - ny_bar, b) * img) / np.sum(img)
    print(m)
    

img = cv.imread('peppers.bmp', cv.IMREAD_GRAYSCALE)
print("Central Moment")
central_moment(img, 2, 0)
central_moment(img, 1, 1)
central_moment(img, 0, 2)