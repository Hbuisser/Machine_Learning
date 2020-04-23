import numpy as np

def sigmoid_(x):
    return 1 / (1 + np.exp(-x))

x = -4
print(sigmoid_(x))


def sigmoid_(x):
    if (type(x) != list):
        return (1 / (1 + np.exp(x * -1)))
    else:
        lst = []
        for xi in x:
            lst.append(1 / (1 + np.exp(xi * -1)))
        return (lst)