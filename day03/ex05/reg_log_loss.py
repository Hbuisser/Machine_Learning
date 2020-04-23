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

def reg_log_loss_(y_true, y_pred, m, theta, lambda_, eps=1e-15):
    if m == 1:
        return (-(y_true * log(y_pred + eps) + (1 - y_true) * log(1 - y_pred + eps)))
    solution = 0
    for yt, yp in zip(y_true, y_pred):
        solution -= yt * log(yp + eps) + (1 - yt) * \
            log(1 - yp + eps) - lambda_ * np.dot(theta, theta) / m
    return solution / m

# Test n.1
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.0)) 
# 7.233346147374828

# Test n.2
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.5)) 
# 8.898346147374827


# Test n.3
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1))
# 10.563346147374826

# Test n.4
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 1)) 
# 49.346258798303566

# Test n.5
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.3)) 
# 22.86292546497024

# Test n.6
x_new = np.arange(1, 13).reshape((3, 4)) 
y_true = np.array([1, 0, 1])
theta = np.array([-5.2, 2.3, -1.4, 8.9]) 
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(reg_log_loss_(y_true, y_pred, m, theta, 0.9)) 
# 45.56292546497025