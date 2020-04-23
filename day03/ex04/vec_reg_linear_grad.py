import numpy as np

def vec_reg_linear_grad(x, y, theta, lambda_):
    if len(x) == 0:
        return None
    m, n = x.shape
    if (y.shape != (m,) and y.shape != (m, 1)) or (theta.shape != (n,) and theta.shape != (n, 1)):
        return None
    return (np.dot(np.transpose(x), np.dot(x, theta) - y) + lambda_ * theta) / m
    #return (np.dot(np.transpose(x), (np.dot(x, theta) - y)) + lambda_ * theta) / len(x)

X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]]) 

Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,10.5,-6])

print(vec_reg_linear_grad(X, Y, Z, 1))
#array([-192.64285714, 887.5, -679.57142857])

print(vec_reg_linear_grad(X, Y, Z, 0.5))
#array([-192.85714286, 886.75, -679.14285714])

print(vec_reg_linear_grad(X, Y, Z, 0.0))
#array([-193.07142857, 886., -678.71428571])

