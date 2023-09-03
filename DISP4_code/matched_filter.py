import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 20)
factor = 20
h = np.linspace(-1 * factor, 1 * factor, 20)
plt.stem(n,h)
plt.show()

n = np.arange(0, 140)
x = np.zeros(140)

factor2 = 15
x[(n > 5) & (n < 26)] = 1
x[(n > 45) & (n < 66)] = np.linspace(-1, 1, 20)
x[(n > 85) & (n < 106)] = -np.linspace(-1, 1, 20)
x[(n > 115) & (n < 136)] = np.sin(np.linspace(0, 1, 20) * 2 * np.pi)
x *= factor2
plt.stem(n, x)
plt.show()

y = np.zeros(len(x) + len(h) - 1)


for i in np.arange(-len(h), len(x)):
    X = np.zeros(len(h))
    tmp = x[max(i, 0): min(i + len(h), len(x))]
    # print(i, tmp)
    if i < 0:
        X[len(h) - len(tmp):] = tmp
    else:
        X[:len(tmp)] = tmp
    if np.all(X == 0):
        y[i] = 0
    else:
        y[i] = np.sum(X * h) / np.sqrt(np.sum(np.square(X)) * np.sum(np.square(h)))

plt.stem(np.arange(1, len(y) + 1), y)
plt.xlabel('n')
plt.ylabel('similarity')
plt.title('Normalization Matched Filter')
plt.show()

y = np.zeros(len(x) + len(h) - 1)


for i in np.arange(-len(h), len(x)):
    X = np.zeros(len(h))
    tmp = x[max(i, 0): min(i + len(h), len(x))]
    # print(i, tmp)
    if i < 0:
        X[len(h) - len(tmp):] = tmp
    else:
        X[:len(tmp)] = tmp

    h = h - np.mean(h)
    if np.all((X - np.mean(X)) == 0):
        y[i] = 0
    else:
        y[i] = np.sum(X * h) / np.sqrt(np.sum(np.square(X - np.mean(X))) * np.sum(np.square(h)))

plt.stem(np.arange(1, len(y) + 1), y)
plt.xlabel('n')
plt.ylabel('similarity')
plt.title('Normalization & Offset Matched Filter')
plt.show()
