from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn import model_selection


def get_dataset(n_samples=100, random_state=42):
    return datasets.make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                        n_classes=2, n_redundant=0,
                                        n_clusters_per_class=2, random_state=random_state)


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
        # if NodeClassificator.LEAFS < self.max_leaf and NodeClassificator.NODES < self.max_deep:
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
        return entropy(y) - p * entropy(y[lhs]) - (1 - p) * entropy(y[rhs])

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


class RandomForestClassificator:

    def get_bootstrap(self, data, labels, n_trees):
        n_samples = data.shape[0]
        bootstrap = []

        data_set = set(range(data.shape[0]))
        out_indexes = []

        for i in range(n_trees):
            b_data = np.zeros(data.shape)
            b_labels = np.zeros(labels.shape)
            b_data_set = set()
            for j in range(n_samples):
                sample_index = np.random.randint(0, n_samples - 1)
                b_data_set.add(sample_index)
                b_data[j] = data[sample_index]
                b_labels[j] = labels[sample_index]
            out_indexes.append(data_set.difference(b_data_set))
            bootstrap.append((b_data, b_labels))
        self.bootstrap = bootstrap
        self.out_indexes = out_indexes

    def fit(self, data, labels, n_trees, min_leaf, max_leaf, max_deep):
        self.get_bootstrap(data, labels, n_trees)
        self.forest = []
        data_set = set()
        for b_data, b_labels in self.bootstrap:
            self.forest.append(DecisionTreeClassificator().fit(b_data, b_labels, min_leaf, max_leaf, max_deep))
        self.oob = self.__get_oob(data, labels)
        return self

    def __get_oob(self, data, labels):
        y_pred = []
        y_i = []
        for i, x in enumerate(data):
            res_x = []
            for out_i in self.out_indexes:
                if i in out_i:
                    label_pred = self.predict([x])[0]
                    res_x.append(label_pred)
            if res_x:
                y_pred.append(Counter(res_x).most_common(1)[0][0])
                y_i.append(i)
        return accuracy(labels[y_i], y_pred)

    def predict(self, data):
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(data))

        predictions_per_object = list(zip(*predictions))

        voted_predictions = []
        for obj in predictions_per_object:
            voted_predictions.append(max(set(obj), key=obj.count))

        return voted_predictions


if __name__ == '__main__':
    train_data, train_labels = get_dataset()

    classifier = RandomForestClassificator().fit(train_data, train_labels, n_trees=50,
                                                 min_leaf=5, max_leaf=100, max_deep=50)

    print(classifier.oob)
'''
    test_data, test_labels = train_data[classifier.out_indexes], train_labels[classifier.out_indexes]
    y_pred = classifier.predict(train_data)
    print(f'Accuracy for train data = {accuracy(train_labels, y_pred)}')
    y_pred = classifier.predict(test_data)
    print(f'Accuracy for test data = {accuracy(test_labels, y_pred)}')

    print('Классификация на 3 деревьях')
    classifier = RandomForestClassificator().fit(train_data, train_labels, n_trees=3,
                                                 min_leaf=5, max_leaf=100, max_deep=50)
    test_data, test_labels = train_data[classifier.out_indexes], train_labels[classifier.out_indexes]
    y_pred = classifier.predict(train_data)
    print(f'Accuracy for train data = {accuracy(train_labels, y_pred)}')
    y_pred = classifier.predict(test_data)
    print(f'Accuracy for test data = {accuracy(test_labels, y_pred)}')

    print('Классификация на 10 деревьях')
    classifier = RandomForestClassificator().fit(train_data, train_labels, n_trees=10,
                                                 min_leaf=5, max_leaf=100, max_deep=50)
    test_data, test_labels = train_data[classifier.out_indexes], train_labels[classifier.out_indexes]
    y_pred = classifier.predict(train_data)
    print(f'Accuracy for train data = {accuracy(train_labels, y_pred)}')
    y_pred = classifier.predict(test_data)
    print(f'Accuracy for test data = {accuracy(test_labels, y_pred)}')

    print('Классификация на 50 деревьях')
    classifier = RandomForestClassificator().fit(train_data, train_labels, n_trees=50,
                                               min_leaf=5, max_leaf=100, max_deep=50)
    test_data, test_labels = train_data[classifier.out_indexes], train_labels[classifier.out_indexes]
    y_pred = classifier.predict(train_data)
    print(f'Accuracy for train data = {accuracy(train_labels, y_pred)}')
    y_pred = classifier.predict(test_data)
    print(f'Accuracy for test data = {accuracy(test_labels, y_pred)}')
'''
