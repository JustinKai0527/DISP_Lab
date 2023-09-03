import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-30, 101)
x = np.zeros(len(n))
x[(-10 < n) & (n < 20) | (50 < n) & (n < 80)] = 1
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Signal Plot')
plt.show()

an =0.4
x += an * (np.random.rand(131) - 0.5)

plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Signal Plot')
plt.show()

# doing edge detection filter
sigma = 0.2
N = np.arange(-5, 6)  # |n| <= 5
C = 1 / np.sum(np.exp(sigma * abs(N)))
sgn = np.zeros(len(N))
sgn[:int(len(N)/2)] = 1
sgn[int(len(N)/2+1):] = -1
# print(sgn)
h = C * sgn * np.exp(-sigma * abs(N))

y = np.convolve(x, h)
plt.stem(np.arange(-30, -30 + len(y)), y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Edge Detection Filter')
plt.show()


an = 0.2
x = np.zeros(len(n))
x[(-10 < n) & (n < 20) | (50 < n) & (n < 80)] = 1
x += an * (np.random.rand(131) - 0.5)

plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Signal Plot')
plt.show()

# doing edge detection filter
sigma = 5
N = np.arange(-5, 6)  # |n| <= 5
C = 1 / np.sum(np.exp(sigma * abs(N)))
sgn = np.zeros(len(N))
sgn[:int(len(N)/2)] = 1
sgn[int(len(N)/2+1):] = -1
# print(sgn)
h = C * sgn * np.exp(-sigma * abs(N))

y = np.convolve(x, h)
n = np.arange(-30, -30 + len(y))
plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Edge Detection Filter')
plt.show()