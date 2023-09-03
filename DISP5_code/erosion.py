import numpy as np
import cv2
    
def dilation(img):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    img = cv2.filter2D(img.astype(np.float32), -1, kernel)  # Convert to float32
    img = np.where(img == 255 * 5, 255, 0).astype(np.uint8)  # Convert back to uint8
    
    return img
    
def erosion(img):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    result = cv2.filter2D(img.astype(np.float32), -1, kernel)  # Convert to float32
    img = np.where(result >= 255, 255, 0).astype(np.uint8)  # Convert back to uint8
    return img

def hole_filling(img, num_ero, num_dila):
    
    for _ in range(num_dila):
        img = dilation(img)
        
    for _ in range(num_ero):
        img = erosion(img)
        
    return img

def opening(img, num_ero, num_dila):
    
    for _ in range(num_ero):
        img = erosion(img)
        
    for _ in range(num_dila):
        img = dilation(img)
        
    return img


img = cv2.imread("peppers.bmp", cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('raw img', img)
cv2.waitKey()

# doing the erosion
erosion_img = erosion(img)
cv2.imshow('erosion img', erosion_img)
cv2.imwrite("erosion_img.png", erosion_img)
cv2.waitKey()

# doing the dilation
dilation_img = dilation(img)
cv2.imshow('dilation img', dilation_img)
cv2.imwrite("dilation_img.png", dilation_img)
cv2.waitKey()

# # doing the hole_filling
img2 = img

hole_filling_img = hole_filling(img2, 3, 3)
cv2.imshow('hole filling img', hole_filling_img)
cv2.waitKey()

# # doing the opening
img3 = img

opening_img = opening(img3, 3, 3)
cv2.imshow('opening img', opening_img)
cv2.waitKey()
