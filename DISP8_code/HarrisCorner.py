import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def CornerDetection(img_file):
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)

    # step 1: doing the difference
    k = np.array([[0,0,0],
                [1,0,-1],
                [0,0,0]])

    I_x = convolve2d(img, k, mode='same')
    # print(I_x[:4, :4])
    k = np.array([[0,1,0],
                [0,0,0],
                [0,-1,0]])

    I_y = convolve2d(img, k, mode='same')

    # step 2: compute the M  = | A C |
    #                          | C B |

    x = np.linspace(-10, 10, 21)
    x, y = np.meshgrid(x, x)

    tau = 3      # choose 1 ~ 5
    w = np.exp(-(x ** 2 + y ** 2) / (2 * (tau ** 2)))  # gaussian filter


    I_x = I_x.astype(np.float32)
    I_y = I_y.astype(np.float32)

    Sxx = I_x * I_x 
    Syy = I_y * I_y
    Sxy = I_x * I_y

    A = convolve2d(Sxx, w, mode='same')
    B = convolve2d(Syy, w, mode='same')
    C = convolve2d(Sxy, w, mode='same')


    # step 3 reponse R = Det(M) - K Trace(M) ** 2
    k = 0.04
    R = (A * B - (C ** 2)) - k * ((A + B) ** 2)

    # step 4 setup threshold 
    # (i) R(m, n) > R(m+a, n+b)       -1<=a,b<=1
    # (ii) R(m, n) > threshold = max(R) / 100

    # Define the range of a and b
    a_range = np.arange(-1, 2)
    b_range = np.arange(-1, 2)
    max = np.max(R)
    # print(max)

    # Create a mask for the condition
    # how to do max in every 3x3 kernel
    # using the roll we can direct do the max in every pos in -1 <= (a, b) <= 1

    R_pad = np.pad(R, ((1,1), (1,1)), mode='constant')
    condition_mask = np.ones_like(R_pad)
    for a in a_range:
        for b in b_range:
            if a == 0 and b == 0:
                continue  # Skip the center element comparison (m, n) vs. (m, n)
            
            condition_mask *= R_pad >= np.roll(np.roll(R_pad, -a, axis=0), -b, axis=1)
            
    condition_mask = condition_mask[1:-1, 1:-1]

    mask2 = np.ones_like(R)
    mask2[R < max / 100] = 0

    condition_mask = condition_mask * mask2
    x, y = np.where(condition_mask == 1)


    img = cv.imread(img_file, cv.IMREAD_COLOR)
    plt.imshow(img)
    plt.scatter(y, x, marker='*', color='red', s=4)

    plt.show()


if __name__ == "__main__":
    CornerDetection('peppers.bmp')
    CornerDetection('tower.BMP')
    CornerDetection('test1.jpg')
    CornerDetection('building.png')