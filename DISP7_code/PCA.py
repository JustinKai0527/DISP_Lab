import numpy as np

# (2, -1, 3),   (-1, 3, 5),   (0, 2, 4),  (4, -2, -1),  (1, 0, 4), (-2, 5, 5)

data = np.array([[2, -1, 3],
                 [-1, 3, 5],
                 [0, 2, 4],
                 [4, -2, -1],
                 [1, 0, 4],
                 [-2, 5, 5]]).astype('float64')

print("mean", np.mean(data, axis=0), sep=" ")
mean = np.mean(data, axis=0)
data -= mean

U, S, Vh = np.linalg.svd(data, full_matrices=False)

print("Dim Redution to 2", U[:, :2], sep='\n')
print("Dim Redution to 1", U[:, 0], sep='\n')
# print(data.dot(Vh[:2, :].T) / S[:2].reshape(1, -1))


# reconstruct the data
print("Reconstruct the data")
data_pca = U[:, 0]
print(S[0] * data_pca.reshape(-1, 1).dot(Vh[0, :].reshape(1, -1)) + mean.reshape(1, -1))
data_pca = U[:, :2]
S1 = np.diag(S[:2])
print(data_pca.dot(S1.dot(Vh[:2, :])) + mean.reshape(1, -1))
