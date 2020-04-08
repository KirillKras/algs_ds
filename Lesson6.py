from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn import model_selection


def get_dataset(random_state=42):
    return datasets.make_regression(n_samples=1000, n_features=2, n_informative=2,
                                    random_state=random_state)


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
        self.lhs = NodeRegressor(self.x, self.y, self.idxs[lhs], tree)
        self.rhs = NodeRegressor(self.x, self.y, self.idxs[rhs], tree)

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
        return Counter(self.y[self.idxs]).most_common(1)[0][0]

    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        y_ = y[lhs]
        p = y[lhs].shape[0] / y.shape[0]
        return gini(y) - p * gini(y[lhs]) - (1-p) * gini(y[rhs])


class NodeRegressor(Node):

    def _val(self):
        return np.mean(self.y[self.idxs])

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
        self.dtree = Node

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


class GBoost:

    def __init__(self, n_trees, max_depth, coefs, eta, stohastic=1.0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.coefs = coefs
        self.eta = eta
        self.stohastic = stohastic
        self.trees = []
        self.train_errors = []
        self.test_errors = []

    def fit_predict(self, X_train, X_test, y_train, y_test):

        for i in range(self.n_trees):

            indx_train = np.random.randint(0, X_train.shape[0],
                                           int(X_train.shape[0] * self.stohastic))
            indx_test = np.random.randint(0, X_test.shape[0],
                                          int(X_test.shape[0] * self.stohastic))
            X_train_st = X_train[indx_train, :]
            X_test_st = X_test[indx_test, :]
            y_train_st = y_train[indx_train]
            y_test_st = y_test[indx_test]

            tree = DecisionTreeRegressor(min_leaf=5, max_leafs=1000, max_nodes=max_depth)
            if len(self.trees) == 0:
                tree.fit(X_train_st, y_train_st)
                self.train_errors.append(calc_mse(y_train_st, self.gb_predict(X_train_st)))
                self.test_errors.append(calc_mse(y_test_st, self.gb_predict(X_test_st)))
            else:
                target = self.gb_predict(X_train_st)
                tree.fit(X_train_st, bias(y_train_st, target))
                self.train_errors.append(calc_mse(y_train_st, self.gb_predict(X_train_st)))
                self.test_errors.append(calc_mse(y_test_st, self.gb_predict(X_test_st)))
            self.trees.append(tree)

    def gb_predict(self, X):
        return np.array([sum([self.eta * coef * alg.predict([x])[0] for alg, coef in zip(self.trees, self.coefs)]) for x in X])


def evaluate_alg(X_train, X_test, y_train, y_test, boost):
    train_prediction = boost.gb_predict(X_train)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} ' +
    f'с шагом {eta} на тренировочной выборке: {calc_mse(y_train, train_prediction)}')

    test_prediction = boost.gb_predict(X_test)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} ' +
    f'с шагом {eta} на тестовой выборке: {calc_mse(y_test, test_prediction)}')


if __name__ == '__main__':
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    n_trees = 50
    coefs = [1] * n_trees
    max_depth = 10
    eta = 0.1

    boost = GBoost(n_trees, max_depth, coefs, eta, stohastic=0.5)
    boost.fit_predict(X_train, X_test, y_train, y_test)

    evaluate_alg(X_train, X_test, y_train, y_test, boost)
