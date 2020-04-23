import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_score_(y_true, y_pred):
    i = 0
    for x, y in zip(y_true, y_pred):
        if x == y:
            i += 1
    return i / len(y_true)


# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1]) 
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0]) 
print(accuracy_score_(y_true, y_pred)) 
print(accuracy_score(y_true, y_pred))
# 0.5 
# 0.5

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(accuracy_score_(y_true, y_pred)) 
print(accuracy_score(y_true, y_pred)) 
# 0.625