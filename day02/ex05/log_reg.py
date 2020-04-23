import pandas as pd
import numpy as np

class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'): 
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.thetas = []

    def gradient_function(self, X, y):
        diff = np.dot(X, self.theta) - y
        return (1./len(X)) * np.dot(np.transpose(X), diff)

    def fit(self, x_train, y_train): 
        X_b = np.c_[np.zeros((len(x_train), 1)) + 1, x_train]
        gradient = self.gradient_function(X_b, y_train)
        for i in range(self.max_iter):
            self.theta = self.theta - alpha * gradient
            gradient = self.gradient_function(X_b, y_train) / 2
        return (self.theta)

    def predict(self, x_train): 
        b = []
        l = len(x_train)
        b = np.zeros((l, 1)) + 1
        new_table = np.c_[b, x_train]
        res = new_table.dot(self.theta)
        return res

    def score(self, x_train, y_train):
        y_pred = self.predict(x)
        score = 0
        return (y_pred == y).mean()



df_train = pd.read_csv('train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0] 

df_test = pd.read_csv('test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]

model = LogisticRegressionBatchGd(alpha=0.01, max_iter=1500, verbose=True, learning_rate='constant')
model.fit(x_train, y_train)

print(f'Score on train dataset : {model.score(x_train, y_train)}') 
y_pred = model.predict(x_test)
print(f'Score on test dataset : {(y_pred == y_test).mean()}')