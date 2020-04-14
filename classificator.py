from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import datasets


def calc_mse(y, y_pred):
  return np.mean((y - y_pred)**2)


def gini(labels):
    classes = Counter(labels)
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2
    return impurity


def entropy(labels):
    classes = Counter(labels)
    impurity = 1
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p * np.log2(p) if p else 0
    return impurity


def accuracy(y, y_pred):
    return np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]


def bias(y, z):
    return (y - z)


class Node(ABC):

    def __init__(self, x, y, idxs, tree):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = tree.min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.score = 0
        self.val = self._val()
        if tree.max_leafs > tree.leafs and tree.max_nodes > tree.nodes:
            tree.nodes += 1
            self.find_split(tree)

    @abstractmethod
    def _val(self):
        pass

    def find_split(self, tree):
        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            tree.leafs += 1
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = self._create_node(lhs, tree)
        self.rhs = self._create_node(rhs, tree)

    @abstractmethod
    def _create_node(self, inds, tree):
        pass

    def find_better_split(self, var_idx):
        x = self.x[self.idxs, var_idx]
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf:
                continue
            curr_score = self.find_score(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    @abstractmethod
    def find_score(self, lhs, rhs):
        pass

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




class NodeClassificator(Node):

    def _val(self):
        prediction = Counter(self.y[self.idxs]).most_common(1)[0][0]
        return prediction

    def _create_node(self, inds, tree):
        return NodeClassificator(self.x, self.y, self.idxs[inds], tree)

    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        y_ = y[lhs]
        p = y[lhs].shape[0] / y.shape[0]
        return gini(y) - p * gini(y[lhs]) - (1-p) * gini(y[rhs])


class NodeRegressor(Node):

    def _val(self):
        return np.mean(self.y[self.idxs])

    def _create_node(self, inds, tree):
        return NodeRegressor(self.x, self.y, self.idxs[inds], tree)

    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        y_std = y.std()
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return y_std * self.idxs.sum() - lhs_std * lhs.sum() - rhs_std * rhs.sum()


class DecisionTree(ABC):

    def __init__(self, min_leaf, max_leafs, max_nodes):
        self.min_leaf = min_leaf
        self.max_leafs = max_leafs
        self.max_nodes = max_nodes
        self.nodes = 0
        self.leafs = 0
        self.dtree = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.dtree.predict(X)


class DecisionTreeClassificator(DecisionTree):

    def fit(self, X, y):
        self.dtree = NodeClassificator(X, y, np.array(np.arange(len(y))), tree=self)
        return self


class DecisionTreeRegressor(DecisionTree):

    def fit(self, X, y):
        self.dtree = NodeRegressor(X, y, np.array(np.arange(len(y))), tree=self)
        return self

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Для начала отмасштабируем выборку
    X_ = X.astype(float)

    rows, cols = X_.shape

    # центрирование - вычитание из каждого значения среднего по строке
    means = X_.mean(0)
    for i in range(rows):
        for j in range(cols):
            X_[i, j] -= means[j]

    # деление каждого значения на стандартное отклонение
    std = np.std(X_, axis=0)
    for i in range(cols):
        for j in range(rows):
            X_[j][i] /= std[i]

    model = DecisionTreeClassificator(min_leaf=5, max_leafs=30, max_nodes=30)
    model.fit(X_, y)
    y_pred = model.predict(X_)
    pass