import math as mt
import numpy as np

def entropy(array):
    N = len(array)
    result = 0.0
    for comp in np.unique(array):
        n = 0
        for item in array:
            if comp == item:
                n += 1
        result -= (n / N) * mt.log2(n / N)
    return result


X = np.array(['0', '0', '1', '0', 'bob', '1'])
print(entropy(X))

