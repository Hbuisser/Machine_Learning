import numpy as np

def gradient_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1./len(X)) * np.dot(np.transpose(X), diff)

def fit_(theta, X, y, alpha, n_cycle):
    X_b = np.c_[np.zeros((len(X), 1)) + 1, X]
    gradient = gradient_function(theta, X_b, y)
    for i in range(n_cycle):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X_b, y) / 2
    return (theta)

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
print(theta1)

