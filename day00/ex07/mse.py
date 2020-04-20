import numpy as np

def mse(y, y_hat):
    solution = 0
    for i, j in zip(y, y_hat):
        solution += (j - i) ** 2
    return 1 / len(y) * solution

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
res = mse(X, Y)
print(res)


