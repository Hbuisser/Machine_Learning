import numpy as np 

def linear_mse(x, y, theta):
    solution = 0
    x = np.dot(x, theta)
    for i, j in zip(x, y):
        solution += (j - i) ** 2
    return 1 / len(y) * solution

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
res = linear_mse(X, Y, Z)
print (res)