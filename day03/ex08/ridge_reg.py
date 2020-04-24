import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge


class MyLR():
    def __init__(self, theta):
        self.theta = theta

    def predict_(self, X):
        b = []
        l = len(X)
        b = np.zeros((l, 1)) + 1
        new_table = np.c_[b, X]
        res = new_table.dot(self.theta)
        return res

    def cost_elem_(self, X, Y):
        solution = []
        l = len(X)
        b = np.zeros((l, 1)) + 1
        X = np.c_[b, X]
        X = np.dot(X, self.theta)
        for i, j in zip(X, Y):
            solution.append((j - i) ** 2)
        solution = np.array(solution)
        return (0.5 / len(Y) * solution)

    def cost_(self, X, Y):
        solution = 0
        l = len(X)
        b = np.zeros((l, 1)) + 1
        X = np.c_[b, X]
        X = np.dot(X, self.theta)
        for i, j in zip(X, Y):
            solution += (j - i) ** 2
        return float(0.5 / len(Y) * solution)

    def gradient_function(self, X, y):
        diff = np.dot(X, self.theta) - y
        return (1./len(X)) * np.dot(np.transpose(X), diff)

    def fit_(self, X, y, alpha, n_cycle):
        X_b = np.c_[np.zeros((len(X), 1)) + 1, X]
        gradient = self.gradient_function(X_b, y)
        for i in range(n_cycle):
            self.theta = self.theta - alpha * gradient
            gradient = self.gradient_function(X_b, y) / 2
        return (self.theta)

    def mse_(self, x, y):
        y_hat = self.predict_(x)
        solution = 0
        for i, j in zip(y, y_hat):
            solution += (j - i) ** 2
        return float(1 / len(y) * solution)

class MyRidge(MyLR):
    def __init__(self, theta):
        self.theta = theta
    def get_params_(self):
        return getattr(self)
    def set_params_(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)
    def predict_(self):
        pass
    def fit_(self):
        pass
    def mse_(self):
        pass
    def rmse_(self):
        pass
    def rscore_(self):
        pass
    # def fit_(self, lambda=1.0, max_iter=1000, tol=0.001):
    #     pass


data = pd.read_csv("data.csv")
loo = LeaveOneOut()
for train, test in loo.split(data):
    print("%s %s" % (train, test))
    
# loo.get_n_splits(data)
# print(loo)

r = Ridge(data)
# print(r.get_params())
#print(r.set_params())
    