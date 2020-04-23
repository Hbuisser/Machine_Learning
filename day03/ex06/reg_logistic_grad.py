import numpy as np
from math import log

def sigmoid_(x):
    if (type(x) != list and type(x) != np.ndarray):
        return (1 / (1 + np.exp(x * -1)))
    else:
        lst = []
        for xi in x:
            lst.append(1 / (1 + np.exp(xi * -1)))
        return (lst)


def reg_logistic_grad(y, x, theta, lambda_):
    return (np.dot(np.transpose(x), sigmoid_(np.dot(x, theta)) - y) + lambda_ * theta) / len(x)


X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]]) 

Y = np.array([1,0,1,1,1,0,0]) 
Z = np.array([1.2,0.5,-0.32])

print(reg_logistic_grad(Y, X, Z, 1))
#array([ 6.69780169, -0.33235792, 2.71787754])

print(reg_logistic_grad(Y, X, Z, 0.5))
#array([ 6.61208741, -0.3680722, 2.74073468]) 

print(reg_logistic_grad(Y, X, Z, 0.0))
#array([ 6.52637312, -0.40378649, 2.76359183])