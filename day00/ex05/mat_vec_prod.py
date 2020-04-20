import numpy as np

def dot(x, y):
    return (float(sum(x_i * y_i for x_i, y_i in zip(x, y))))

def mat_vec_prod(x, y):
    lst = []
    for i in x:
        lst.append(dot(i, y))
    return lst





x = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14], [-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13], [ 2, -1, 12, 3, -7, -3, -6]
])

y = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
res = mat_vec_prod(x, y)
print(res)