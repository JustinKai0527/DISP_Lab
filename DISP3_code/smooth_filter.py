import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-50, 101)
x = 0.1 * n
an = 1
noise = an * (np.random.rand(151) - 0.5)
x += noise

plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Signal Plot')
plt.show()

N = np.arange(-5,6)
sigma = 1
C = 1 / np.sum(np.exp(-sigma * abs(N)))
h = C * np.exp(-sigma * abs(N))

y = np.convolve(x, h)
plt.stem(np.arange(-50, -50 + len(y)), y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Edge Detection Filter')
plt.show()


n = np.arange(-50, 101)
x = 0.1 * n
an = 1.5
noise = an * (np.random.rand(151) - 0.5)
x += noise

plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Signal Plot')
plt.show()

N = np.arange(-5,6)
sigma = 10
C = 1 / np.sum(np.exp(-sigma * abs(N)))
h = C * np.exp(-sigma * abs(N))

y = np.convolve(x, h)
plt.stem(np.arange(-50, -50 + len(y)), y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Edge Detection Filter')
plt.show()





