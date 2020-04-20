import numpy as np

def sum_(x, f):
    a = 0.0
    lst = []
    for i in x:
        lst.append(f(i))
    for i in lst:
        a = a + i
    return a

X = np.array([0, 15, -9, 7, 12, 3, -21])
y = sum_(X, lambda x: x)
print(y)

X = np.array([0, 15, -9, 7, 12, 3, -21])
y = sum_(X, lambda x: x**2)
print(y)