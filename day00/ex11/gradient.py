import numpy as np

def gradient(x, y, theta):
    tot = []
    for xi, yi in zip(x, y):
        theta_xi = float(np.dot(theta, xi))
        tot.append(np.dot((theta_xi - yi), xi))
    tot = np.array(tot)
    return np.sum(tot, axis=0) / len(x)

    # if len(x) == 0:
    #     return None
    # m, n = x.shape
    # if y.shape != (m,) or theta.shape != (n,):
    #     return None
    # res = np.zeros((n,))
    # for i in range(m):
    #     tx = np.dot(theta, x[i])
    #     res = np.add(res, np.dot((tx - y[i]), x[i]))
    # return res / len(x)

X = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]
])

Y = np.array([2, 14, -13, 5, 12, 4, -19]) 
Z = np.array([3,0.5,-6])
res = gradient(X, Y, Z)
print(res)