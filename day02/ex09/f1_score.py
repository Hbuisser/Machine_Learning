import numpy as np
from sklearn.metrics import f1_score

def recall_score_(y_true, y_pred, pos_label=1):
    i = 0
    j = 0
    for x in y_true:
        if x == pos_label:
            j += 1
    for x, y in zip(y_true, y_pred):
        if x == y and x == pos_label:
            i += 1
    return i / j

def precision_score_(y_true, y_pred, pos_label=1):
    i = 0
    j = 0
    for x in y_pred:
        if x == pos_label:
            j += 1
    for x, y in zip(y_true, y_pred):
        if x == y and x == pos_label:
            i += 1
    return i / j

def f1_score_(y_true, y_pred, pos_label=1):
    return 2 * (precision_score_(y_true, y_pred, pos_label) * recall_score_(y_true, y_pred, pos_label)) / (precision_score_(y_true, y_pred, pos_label) + recall_score_(y_true, y_pred, pos_label))


# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1]) 
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0]) 
print(f1_score_(y_true, y_pred)) 
print(f1_score(y_true, y_pred))
# 0.5 
# 0.5

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(f1_score_(y_true, y_pred, pos_label='dog')) 
print(f1_score(y_true, y_pred, pos_label='dog')) 
# 0.6666666666666665
# 0.6666666666666665

# Test n.3
print(f1_score_(y_true, y_pred, pos_label='norminet')) 
print(f1_score(y_true, y_pred, pos_label='norminet')) 
# 0.5714285714285715
# 0.5714285714285715