import numpy as np

n = np.arange(0, 11)

v1 = np.ones_like(n)
v = list()
v.append(v1)

for i in np.arange(1, 5):
    v.append(np.power(n, i))

# print(v)
# using gram-schmidt transform v into orthonormal vec

basis = list()

for vec in v:
    for u in basis:
        vec = vec - np.dot(u, vec) * u
    basis.append(vec / np.linalg.norm(vec))


print("Checking each is orthogonal")
for (i, vec) in enumerate(basis):
    print(i)
    for j, u in enumerate(basis):
        
        if j != i:
            print((vec * u).sum(), sep=' ')


for v in basis:
    print(v)
    
    