import matplotlib.pyplot as plt
import numpy as np

# y(t) = (2t)^2        0 < t < 1
# y(t) = (4 - 2t)^2    1 < t < 2

ts = 0.1

n = np.arange(0,2.1,ts)
x_n = np.zeros(21)
x_n[:10] = np.square(2 * n[:10])
x_n[10:] = np.square(4 - 2 * n[10:])
n = np.arange(0,21)
plt.plot(n, x_n, marker='o', color='black')
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("x[n] vs n")
plt.show()

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

X = DFT(x_n)
tmp = np.zeros(len(X))
tmp[:10] = X[11:]
tmp[10:] = X[:11]
print(tmp[10:10])
Y = tmp
n = np.arange(-5,5.5,0.5)
plt.plot(n, abs(Y), marker='o', color='black')
plt.xlabel("w")
plt.ylabel("Y(w)")
plt.show()


