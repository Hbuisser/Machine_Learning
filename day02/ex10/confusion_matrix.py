import numpy as np
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_true, y_pred, labels=None):    
    all_labels = np.unique(np.c_[y_pred, y_true])
    if labels:
        all_labels = labels
    all_labels.sort()
    new = np.zeros((len(all_labels), len(all_labels)))
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            for yt, yp in zip(y_true, y_pred):
                if (all_labels[i] == yt and all_labels[j] == yp):
                    new[i][j] += 1
    return new.astype(int)




y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet']) 
print(confusion_matrix_(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
# [[0 0 0]
# [0 2 1]
# [1 0 2]]
# [[0 0 0]
# [0 2 1]
# [1 0 2]]
print(confusion_matrix_(y_true, y_pred, labels=['dog', 'norminet'])) 
print(confusion_matrix(y_true, y_pred, labels=['dog', 'norminet']))
# [[2 1]
# [0 2]]
# [[2 1]
# [0 2]]
