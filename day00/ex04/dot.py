import numpy as np

def dot(x, y):
    return (float(sum(x_i * y_i for x_i, y_i in zip(x, y))))

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

result = dot(X, Y)
print(result)