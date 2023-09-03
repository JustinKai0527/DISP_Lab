import numpy as np
import cv2

img1 = cv2.imread('marilyn.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('einstein.bmp', cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

h, w = img1.shape
th = int(max(h, w) / 30)

fourier1 = np.fft.fft2(img1)
fourier2 = np.fft.fft2(img2)

fourier1 = np.fft.fftshift(fourier1)
fourier2 = np.fft.fftshift(fourier2)

hybrid_fourier = np.zeros(fourier1.shape, dtype=complex)

for i in range(h):
    for j in range(w):
        
        distance = np.sqrt((i - h // 2)**2 + (j - w // 2)**2)
        H = 1 if distance <= th else 0
        hybrid_fourier[i, j] = H * fourier1[i, j] + (1 - H) * fourier2[i, j]

hybrid_fourier = np.fft.ifftshift(hybrid_fourier)

hybrid_img = np.real(np.fft.ifft2(hybrid_fourier)).astype(np.uint8)

cv2.imshow('image', hybrid_img)
cv2.imwrite('hybrid.bmp', hybrid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

