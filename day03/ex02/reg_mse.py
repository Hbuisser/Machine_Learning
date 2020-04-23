import numpy as np

def vectorized_regularization(theta, lambda_):
    return np.dot(theta, np.transpose(theta)) * lambda_

def reg_mse(x, y, theta, lambda_):
    reg = vectorized_regularization(theta, lambda_)
    y_hat = np.dot(x, np.transpose(theta))
    solution = 0
    for i, j in zip(y, y_hat):
        solution += (j - i) ** 2 + reg
    return float(1 / len(y) * solution)

# version protegee
    # m, n = x.shape
    # if (y.shape != (m,) and y.shape != (m, 1)) or (theta.shape != (n,) and theta.shape != (n, 1)):
    #     return None
    # xt = np.dot(x, theta)
    # res = np.dot(np.transpose(xt - y), (xt - y)) + \
    #     np.dot(lambda_ * theta, theta)
    # return res / m

X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19]) 
Z = np.array([3,0.5,-6])

res = reg_mse(X, Y, Z, 0)
print(res)
print(reg_mse(X, Y, Z, 0.1))