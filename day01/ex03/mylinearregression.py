import numpy as np

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

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

res1 = mylr.predict_(X)
print(res1)

res2 = mylr.cost_elem_(X,Y)
print(res2)

res3 = mylr.cost_(X,Y)
print(res3)

res4 = mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
#print(res4)

res5 = mylr.theta
print(res5)

res6 = mylr.predict_(X)
print(res6)

res7 = mylr.cost_elem_(X,Y)
print(res7)

res8 = mylr.cost_(X,Y)
print(res8)