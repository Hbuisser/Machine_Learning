import numpy as np

def cost_elem_(theta, X, Y):
    solution = []
    l = len(X)
    b = np.zeros((l, 1)) + 1
    X = np.c_[b, X]
    X = np.dot(X, theta)
    for i, j in zip(X, Y):
        solution.append((j - i) ** 2)
    solution = np.array(solution)
    return (0.5 / len(Y) * solution)

def cost_(theta, X, Y):
    solution = 0
    l = len(X)
    b = np.zeros((l, 1)) + 1
    X = np.c_[b, X]
    X = np.dot(X, theta)
    for i, j in zip(X, Y):
        solution += (j - i) ** 2
    return float(0.5 / len(Y) * solution)

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
res = cost_elem_(theta1, X1, Y1)
print(res)
res2 = cost_(theta1, X1, Y1)
print(res2)