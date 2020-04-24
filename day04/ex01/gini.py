import numpy as np

def gini(array):
    result = 0.0
    for comp in np.unique(array):
        n = 0
        for item in array:
            if item == comp:
                n += 1
        result += (n / len(array)) ** 2
    return 1 - result


X = np.array(['0', '0', '1', '0', 'bob', '1'])
print(gini(X))