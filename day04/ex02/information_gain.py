import numpy as np
import math as mt

def gini(array):
    result = 0.0
    for comp in np.unique(array):
        n = 0
        for item in array:
            if item == comp:
                n += 1
        result += (n / len(array)) ** 2
    return 1 - result

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

def information_gain(array_source, array_children_list, criterion='gini'):
    if criterion == gini:
        return gini(array_source) - gini(array_children_list)
    else:
        return entropy(array_source) - entropy(array_children_list)




AS = np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
AC = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
criterion = 0.4689955935892812
print(information_gain(AS, AC, gini)) 
print(information_gain(AS, AC, criterion)) 
#0.18 



