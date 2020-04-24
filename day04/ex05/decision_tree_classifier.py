from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
import pandas as pd
from node import Node

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=10):
        """
        :param str criterion: 'gini' or 'entropy'
        :param max_depth: max_depth of the tree (Decision tree creation stops splitting a node if node.depth >= max_depth)
        """
        self.root = Node(split_feature=2, split_kind='<=', split_criteria=1.9)

    def fit(self, X, y):
        self.root.data = X
        self.root.y = y
        if (self.root.split_kind == '<='):
            for elem in X[self.root.split_feature]:
                current = self.root
                while current.is_leaf == False:
                    if elem <= current.split_criteria:
                        current = current.left_child
                    else:
                        current = current.right_child
                if current.split_criteria == '<=':
                    current.is_leaf = False
                    if elem <= current.split_criteria:
                        current.left_child = Node(data=X, labels=y, is_leaf=True, split_feature=current.split_feature, split_kind=current.split_kind, split_criteria=elem, left=None, right=None, depth=current.depth + 1)
                    else:
                        current.right_child = Node(data=X, labels=y, is_leaf=True, split_feature=current.split_feature, split_kind=current.split_kind, split_criteria=elem, left=None, right=None, depth=current.depth + 1)
    
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


if __name__ == '__main__':
    iris = load_iris()

    X = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    


    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)

    root = dec_tree.root

    print("TEST ON IRIS DATASET")
    print("Root split info = 'Feature_{}{}{}'\n".format(root.split_feature, root.split_kind, root.split_criteria))
    print("5 first lines of the labels of the left child of root =\n{}\n".format(root.left_child.y.head()))
    print("5 first lines of the labels of the right child of root =\n{}".format(root.right_child.y.head()))