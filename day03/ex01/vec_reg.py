import numpy as np

def vectorized_regularization(theta, lambda_):
    return np.dot(theta, np.transpose(theta)) * lambda_

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(vectorized_regularization(X, 0.3))
print(vectorized_regularization(X, 0.01))
print(vectorized_regularization(X, 0))