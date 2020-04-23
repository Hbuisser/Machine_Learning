import numpy as np

def vec_gradient(x, y, theta):
    m = len(x)
    diff = np.dot(x, theta) - y
    return (1./m) * np.dot(np.transpose(x), diff)

    # if len(x) == 0:
    #     return None
    # m, n = x.shape
    # if y.shape != (m,) or theta.shape != (n,):
    #     return None
    # return np.dot(np.transpose(x), (np.dot(x, theta) - y)) / m


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

res = vec_gradient(X, Y, Z)
print(res)