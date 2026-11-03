import numpy as np

# This function centers the tensor in the first mode.
def centerA(X):
    I, J, K = X.shape
    for k in range(K):
        for j in range(J):
            m = np.mean(X[:, j, k])
            X[:, j, k] -= m

# This function centers the tensor in the second mode.
def centerB(X):
    I, J, K = X.shape
    for k in range(K):
        for i in range(I):
            m = np.mean(X[i, :, k])
            X[i, :, k] -= m

# This function centers the tensor in the third mode.
def centerC(X):
    I, J, K = X.shape
    for i in range(I):
        for j in range(J):
            m = np.mean(X[i, j, :])
            X[i, j, :] -= m

# This function carries out the scaling of the data in the first mode.
def scalingA(X):
    I, J, K = X.shape
    for i in range(I):
        d = np.sqrt(np.sum(X[i, :, :] ** 2) / (J * K))
        X[i, :, :] /= d

# This function carries out the scaling of the data in the second mode.
def scalingB(X):
    I, J, K = X.shape
    for j in range(J):
        d = np.sqrt(np.sum(X[:, j, :] ** 2) / (I * K))
        X[:, j, :] /= d

# This function carries out the scaling of the data in the third mode.
def scalingC(X):
    I, J, K = X.shape
    for k in range(K):
        d = np.sqrt(np.sum(X[:, :, k] ** 2) / (I * J))
        X[:, :, k] /= d
