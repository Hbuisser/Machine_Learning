import numpy as np
from math import log

def sigmoid_(x):
    if (type(x) != list):
        return (1 / (1 + np.exp(x * -1)))
    else:
        lst = []
        for xi in x:
            lst.append(1 / (1 + np.exp(xi * -1)))
        return (lst)

def log_loss_(y_true, y_pred, m, eps=1e-15):
    if m == 1:
        return (-(y_true * log(y_pred + eps) + (1 - y_true) * log(1 - y_pred + eps)))
    solution = 0
    for yt, yp in zip(y_true, y_pred):
        solution -= yt * log(yp + eps) + (1 - yt) * log(1 - yp + eps)
    return solution * 1 / len(y_pred)


x = 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1 # length of y_true is 1 
print(log_loss_(y_true, y_pred, m))

x = [1, 2, 3, 4]
y_true = 0
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = sum([a*b for a, b in zip(x, theta)]) 
y_pred = sigmoid_(x_dot_theta)
m= 1
print(log_loss_(y_true, y_pred, m))

x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] 
y_true = [1, 0, 1]
theta = [-1.5, 2.3, 1.4, 0.7]
x_dot_theta = []

for i in range(len(x_new)): 
    my_sum = 0
    for j in range(len(x_new[i])): 
        my_sum += x_new[i][j] * theta[j]
    x_dot_theta.append(my_sum) 
y_pred = sigmoid_(x_dot_theta)
m = len(y_true)
print(log_loss_(y_true, y_pred, m))