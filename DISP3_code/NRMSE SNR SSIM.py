import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import skimage.measure

# Dark mode Light mode
img1 = cv2.imread('BABOON.BMP', cv2.IMREAD_COLOR)
YCrCb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
YCrCb[:,:,0] = 255 * ((YCrCb[:,:,0] / 255) ** (0.5))
Light_img1 = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
# inverse_img1 = 255 - img1
cv2.imwrite(".png", Light_img1)
# cv2.imshow('image', Light_img1)
# cv2.waitKey(0)

img2 = cv2.imread('peppers256.bmp', cv2.IMREAD_COLOR)

# doing the NRMSE
NRMSE1 = np.sqrt(np.sum(np.square(Light_img1 - img1)) / np.sum(np.square(img1)))
NRMSE2 = np.sqrt(np.sum(np.square(img2 - img1)) / np.sum(np.square(img1)))
print("NRMSE :", NRMSE1, " ", NRMSE2)

# doing the PSNR
PSNR1 = 10 * np.log10(255 ** 2 / np.mean(np.square(Light_img1 - img1)))
PSNR2 = 10 * np.log10(255 ** 2 / np.mean(np.square(img2 - img1)))
print("SNR :", PSNR1, PSNR2)

# doing the SSIM
def ssim(img1, img2):
    mean_x = np.mean(img1)
    mean_y = np.mean(img2)
    var_x = np.var(img1)
    var_y = np.var(img2)
    cov = np.sum((img1 - mean_x) * (img2 - mean_y)) / (len(img1) - 1)
    L = np.max(img1) - np.min(img1)
    c1 = np.sqrt(1/L)
    c2 = c1

    return (((2 * mean_x * mean_y) + (c1 * L) ** 2) * (2 * cov) + (c2 * L) ** 2) / (((mean_x) ** 2 + (mean_y) ** 2 + (c1 * L) ** 2) * ((var_x) ** 2 + (var_y) ** 2 + (c2 * L) ** 2))

print("SSIM :", ssim(img1, Light_img1), ssim(img1, img2))