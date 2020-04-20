import numpy as np

def mean(x):
    a = 0.0
    lst = []
    for i in x:
        lst.append(i)
    for i in lst:
        a = a + i
    return a / len(x)

def variance(x):
    m = mean(x)
    l = len(x)
    var = 0.0
    for i in x:
        var = var + (i - m)**2
    result = var / l
    return result

def std(x):
    return variance(x)**(1/2)

X = np.array([0, 15, -9, 7, 12, 3, -21])
y = std(X)
z = np.std(X)
print(y)
print(z)
