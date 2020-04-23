import numpy as np

def regularization(theta, lambda_):
    res = 0
    for t in theta:
        t = t ** 2
        res = res + t
    return lambda_ * res

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(regularization(X, 0.3))
print(regularization(X, 0.01))