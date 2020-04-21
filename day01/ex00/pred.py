import numpy as np

def predict_(theta, X):
    b = []
    l = len(X)
    # while l > 0:
    #     b.append(1)
    #     l-=1
    b = np.zeros((l, 1)) + 1
    new_table = np.c_[b, X]
    res = new_table.dot(theta)
    return res

# def predict_(theta, X):
#     X[np.shape(X)[1] + 1] = np.zeros((len(X), 1), dtype=int) + 1
#     res = X.dot(theta)
#     return res

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
res = predict_(theta1, X1)
print(res)

X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
res = predict_(theta3, X3)
print(res)