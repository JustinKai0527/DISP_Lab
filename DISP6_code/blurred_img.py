import numpy as np
import cv2 as cv
import scipy

img = cv.imread("tower.BMP", cv.IMREAD_GRAYSCALE)
# print(img.shape)
cv.imshow('img', img)
cv.waitKey()

# create 2d-gaussian distribution
n = np.linspace(-10, 10, 21)

x, y = np.meshgrid(n, n)
kernel = np.exp(-0.1 * (np.square(x) + np.square(y)))
kernel = kernel / np.sum(kernel)

h, w = img.shape
L = 10
k1 = np.zeros((h, w), dtype=complex)

# we want to using fourier to do the convolution so we make the kernel
# to put into the k1(which shape same as the img)
k1[:L+1, :L+1] = kernel[L:, L:]
k1[h - L:, :L+1] = kernel[:L, L:]
k1[:L+1, w - L:] = kernel[L:, :L]
k1[h - L:, w - L:] = kernel[:L, :L]

# create noise
noise = np.random.randn(*img.shape)

# create blurred image
img_fourier = np.fft.fftshift(np.fft.fft2(img))
k1_fourier = np.fft.fftshift(np.fft.fft2(k1))
noise_fourier = np.fft.fftshift(np.fft.fft2(noise))

y_fourier = img_fourier * k1_fourier + noise_fourier

y = np.real(np.fft.ifft2(np.fft.ifftshift(y_fourier))).astype(np.uint8)
cv.imshow('blur img', y)
cv.waitKey()

# equalizer H
C = 0.001
H_fourier = 1 / ((C / np.conjugate(k1_fourier)) + k1_fourier)

# reconstruct input image
x_fourier = H_fourier * y_fourier
x = np.real(np.fft.ifft2(np.fft.ifftshift(x_fourier))).astype(np.uint8)
cv.imshow('reconstruct img', x)
cv.waitKey()


