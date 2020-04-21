import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_squared_error

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

data = pd.read_csv("are_blue_pills_magics.csv")
print(data)
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8.7]])) 
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print(linear_model1.mse_(Xpill, Yscore))
print(mean_squared_error(Yscore, Y_model1))
print(linear_model2.mse_(Xpill, Yscore))
print(mean_squared_error(Yscore, Y_model2))

# first graph
pyplot.scatter(Xpill, Yscore)
#pyplot.plot()
pyplot.plot(Xpill, Y_model1, color='lightblue', linewidth=3)
pyplot.title('are_blue_pills_magics')
pyplot.scatter(Xpill, Y_model1)
pyplot.show()

# second graph
# pyplot.scatter(Xpill, Yscore)
# pyplot.plot()
# pyplot.title('are_blue_pills_magics')
# pyplot.scatter(Xpill, Y_model1)
# pyplot.show()