from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn import model_selection


def get_dataset(random_state=42):
    return datasets.make_classification(n_samples=1000, n_features=2, n_informative=2,
                                        n_classes=2, n_redundant=0,
                                        n_clusters_per_class=2, random_state=random_state)

'''
class Node:

    def __init__(self, x, y, idxs, min_leaf=5):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    def find_varsplit(self):
        for c in range(self.col_count): self.find_better_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)
        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)

    def find_better_split(self, var_idx):

        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue

            curr_score = self.find_score(lhs, rhs)
            if curr_score < self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
'''


def gini(labels):
    classes = Counter(labels)
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2
    return impurity


class NodeClassificator:

    LEAFS: int = 0
    NODES = 0

    def __init__(self, x, y, idxs, min_leaf, max_leaf, max_deep):
        NodeClassificator.NODES += 1
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.max_leaf = max_leaf
        self.max_deep = max_deep
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.score = 0
        self.val = Counter(y[idxs]).most_common(1)[0][0]
        if NodeClassificator.LEAFS < self.max_leaf and NodeClassificator.NODES < self.max_deep:
            self.find_varsplit()

    def find_varsplit(self):
        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            NodeClassificator.LEAFS += 1
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = NodeClassificator(self.x, self.y, self.idxs[lhs], self.min_leaf, self.max_leaf, self.max_deep)
        self.rhs = NodeClassificator(self.x, self.y, self.idxs[rhs], self.min_leaf, self.max_leaf, self.max_deep)

    def find_better_split(self, var_idx):

        x = self.x[self.idxs, var_idx]
        x_values = np.unique(x)
        for r in x_values:
            lhs = x <= r
            rhs = x > r
            if NodeClassificator.LEAFS >= self.max_leaf:
                pass
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf:
                continue
            curr_score = self.find_score(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = r

    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        y_ = y[lhs]
        p = y[lhs].shape[0] / y.shape[0]
        return gini(y) - p * gini(y[lhs]) - (1-p) * gini(y[rhs])

    @property
    def split_col(self):
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)


class DecisionTreeClassificator:

    leafs = 0
    nodes = 0

    def fit(self, X, y, min_leaf, max_leaf, max_deep):
        self.dtree = NodeClassificator(X, y, np.array(np.arange(len(y))), min_leaf, max_leaf, max_deep)
        return self

    def predict(self, X):
        return self.dtree.predict(X)


def accuracy(y, y_pred):
    return np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]


def print_tree(node: NodeClassificator, spacing=""):
    # Если лист, то выводим его прогноз
    if node.is_leaf:
        print(spacing + "Прогноз:", node.val)
        return

    # Выведем значение индекса и порога на этом узле
    print(spacing + 'Индекс', str(node.var_idx))
    print(spacing + 'Порог', str(node.split))

    # Рекурсионный вызов функции на положительном поддереве
    print(spacing + '--> True:')
    print_tree(node.lhs, spacing + "  ")

    # Рекурсионный вызов функции на положительном поддереве
    print(spacing + '--> False:')
    print_tree(node.rhs, spacing + "  ")


if __name__ == '__main__':
    classification_data, classification_labels = get_dataset()

    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(classification_data,
                                                                                        classification_labels,
                                                                                        test_size=0.3,
                                                                                        random_state=42)

    classifier = DecisionTreeClassificator().fit(train_data, train_labels, min_leaf=5, max_leaf=100, max_deep=50)
    print_tree(classifier.dtree)
    y_pred = classifier.predict(train_data)
    print(f'Accuracy for train data = {accuracy(train_labels, y_pred)}')
    y_pred = classifier.predict(test_data)
    print(f'Accuracy for test data = {accuracy(test_labels, y_pred)}')

