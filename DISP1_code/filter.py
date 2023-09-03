import cv2
import numpy

# str = input("")
img = cv2.imread("BABOON.BMP", cv2.IMREAD_GRAYSCALE)

cv2.imshow('image', img)
# print(type(img))  # numpy.ndarray

# horizontal filter
horizontal = numpy.diff(img, axis=0)
cv2.imwrite('horizontal.png', horizontal)
cv2.imshow('image', horizontal)
cv2.waitKey(0)

# vertical filter
vertical = numpy.diff(img, axis=1)
cv2.imwrite('vertical.png', vertical)
cv2.imshow('image', vertical)
cv2.waitKey(0)

# sobel operator (horizontal axis)
kernel = numpy.array([[1,0,-1],
                      [2,0,-2],
                      [1,0,-1]])
kernel = kernel / 4
print(kernel)
sobel_horizontal = cv2.filter2D(img, -1, kernel)
cv2.imwrite('sobel_horizontal.png', sobel_horizontal)
cv2.imshow('image', sobel_horizontal)
cv2.waitKey(0)

# sobel operator (vertical axis)
kernel = numpy.array([[1,2,1],
                      [0,0,0],
                      [-1,-2,-1]])
kernel = kernel / 4
print(kernel)
sobel_vertical = cv2.filter2D(img, -1, kernel)
cv2.imwrite('sobel_vertical.png', sobel_vertical)
cv2.imshow('image', sobel_vertical)
cv2.waitKey(0)

# sobel operator (45 degree axis)
kernel = numpy.array([[0,-1,-2],
                      [1,0,-1],
                      [2,1,0]])
kernel = kernel / 4
print(kernel)
sobel_45degree = cv2.filter2D(img, -1, kernel)
cv2.imwrite("sobel_45degree.png", sobel_45degree)
cv2.imshow('image', sobel_45degree)
cv2.waitKey(0)

# sobel operator (135 degree axis)
kernel = numpy.array([[-2,-1,0],
                      [-1,0,1],
                      [0,1,2]])
kernel = kernel / 4
print(kernel)
sobel_135degree = cv2.filter2D(img, -1, kernel)
cv2.imwrite("sobel_135degree.png", sobel_135degree)
cv2.imshow('image', sobel_135degree)
cv2.waitKey(0)

# Laplacian
kernel = numpy.array([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]])
kernel = kernel / 8
print(kernel)
Laplacian = cv2.filter2D(img, -1, kernel)
cv2.imwrite("Laplacian_operator.png", Laplacian)
cv2.imshow('image', Laplacian)
cv2.waitKey(0)



# Dark mode Light mode
img2 = cv2.imread('019.BMP', cv2.IMREAD_COLOR)
YCrCb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
YCrCb[:,:,0] = 255 * ((YCrCb[:,:,0] / 255) ** (0.1))
Light_img = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
cv2.imwrite("Light_Mode_img.png", Light_img)
cv2.imshow('image', Light_img)
cv2.waitKey(0)

img2 = cv2.imread('019.BMP', cv2.IMREAD_COLOR)
YCrCb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
YCrCb[:,:,0] = 255 * ((YCrCb[:,:,0] / 255) ** (5))
darkimg = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
cv2.imwrite("Dark_Mode_img.png", darkimg)
cv2.imshow('image', darkimg)
cv2.waitKey(0)

