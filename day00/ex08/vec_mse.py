import numpy as np

def vec_mse(y, y_hat):
    MSE = np.square(np.subtract(y, y_hat)).mean() 
    return (MSE)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
res = vec_mse(X, Y)
print(res)



# def vect_mse(y, y_hat):
#     diff = y_hat - y
#     res = np.dot(diff, np.reshape(diff, (len(diff), 1)))
#     return float(1 / len(y) * res)

