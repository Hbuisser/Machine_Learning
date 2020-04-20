import numpy as np

def mean(x, f):
    m = len(x)
    a = 0.0
    lst = []
    for i in x:
        lst.append(f(i))
    for i in lst:
        a = a + i
    result = a / m
    return result

X = np.array([0, 15, -9, 7, 12, 3, -21])
y = mean(X, lambda x: x)
print(y)

def mean(x):
    a = 0.0
    lst = []
    for i in x:
        lst.append(i)
    for i in lst:
        a = a + i
    return a / len(x)

X = np.array([0, 15, -9, 7, 12, 3, -21])
y = mean(X)
print(y)