from utils import timeit
import numpy as np
from statistics import mode


class Node:
    def __init__(
        self,
        split_feature: int | None,
        split_value: float | None,
        level: int,
        classification,
        node_impurity: float,
    ):
        self.split_feature = split_feature
        self.split_value = split_value
        self.level = level
        self.classification = classification
        self.children = []
        self.node_impurity = node_impurity


class ClassificationDecisionTree:
    """
    Balanced binomial tree.

    Continuous variable only. No missing value. Independent variable should be mapped to positive integers.
    """
    def __init__(
        self,
        max_level=None,
        impurity_method="gini"
    ):
        self.max_level = max_level
        self.tree = None
        if impurity_method == 'gini':
            self.impurity_func = self.gini_index
        elif impurity_method == 'entropy':
            self.impurity_func = self.cross_entropy
        elif impurity_method == 'misclassification':
            self.impurity_func = self.misclassification
        else:
            raise NotImplementedError(f'Impurity method {impurity_method} is not implemented.')

    @classmethod
    def __sample_prob(cls, y, sample_weights):
        return np.bincount(y, weights=sample_weights)

    @classmethod
    def gini_index(cls, y, sample_weights):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y, sample_weights)
        gini = sum(p * (1 - p))
        return gini

    @classmethod
    def cross_entropy(cls, y, sample_weights):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y, sample_weights)
        deviance = - sum(p * np.log(p))
        return deviance

    @classmethod
    def misclassification(cls, y, sample_weights):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y, sample_weights)
        return 1 - np.max(p)

    def split_node(self, X, y, sample_weight, feature, val):
        left_child_mask = X[:, feature] <= val
        right_child_mask = (~left_child_mask)

        X_left, X_right = X[left_child_mask], X[right_child_mask]
        y_left, y_right = y[left_child_mask], y[right_child_mask]
        sample_weights_left, sample_weights_right = (
            sample_weight[left_child_mask], sample_weight[right_child_mask]
        )
        weights_sum_left, weights_sum_right = (sum(sample_weights_left), sum(sample_weights_right))
        sample_weights_left, sample_weights_right = (
            sample_weights_left / weights_sum_left, sample_weights_right / weights_sum_right
        )
        return (
            X_left, X_right,
            y_left, y_right,
            sample_weights_left, sample_weights_right,
            weights_sum_left, weights_sum_right
        )

    def build_tree(self, X: np.array, y: np.array, level: int = 0, sample_weight=None):

        n_sample, n_feature = X.shape
        if sample_weight is None:
            sample_weight = np.ones((n_sample,)) / n_sample

        y_count = np.bincount(y, weights=sample_weight)
        current_node_impurity = self.impurity_func(y, sample_weight)
        if (self.max_level is not None and level >= self.max_level) or len(np.unique(y)) <= 1:
            return Node(
                split_feature=None,
                split_value=None,
                level=level,
                classification=np.argmax(y_count),
                node_impurity=current_node_impurity,
            )

        feature_impurity = dict()
        feature_val = dict()

        for feature in range(n_feature):

            min_node_impurity, split_val = current_node_impurity, None

            for val in set(X[:, feature]):

                (
                    X_left, X_right,
                    y_left, y_right,
                    sample_weights_left, sample_weights_right,
                    weights_sum_left, weights_sum_right
                ) = self.split_node(X, y, sample_weight, feature, val)

                node_impurity = (
                    weights_sum_left * self.impurity_func(y_left, sample_weights_left)
                    + weights_sum_right * self.impurity_func(y_right, sample_weights_right)
                )

                if node_impurity < min_node_impurity:
                    min_node_impurity = node_impurity
                    split_val = val

            feature_impurity[feature] = min_node_impurity
            feature_val[feature] = split_val

        split_feature = min(feature_impurity, key=feature_impurity.get)
        split_val, node_impurity = feature_val[split_feature], feature_impurity[split_feature]

        if split_val is None:
            return Node(
                split_feature=None,
                split_value=None,
                level=level,
                classification=np.argmax(y_count),
                node_impurity=node_impurity,
            )

        (
            X_left, X_right,
            y_left, y_right,
            sample_weights_left, sample_weights_right,
            weights_sum_left, weights_sum_right
        ) = self.split_node(X, y, sample_weight, split_feature, split_val)

        node = Node(
            split_feature=split_feature,
            split_value=split_val,
            level=level,
            classification=np.argmax(y_count),
            node_impurity=node_impurity,
        )

        left_child = self.build_tree(X_left, y_left, level=level + 1, sample_weight=sample_weights_left)
        right_child = self.build_tree(X_right, y_right, level=level + 1, sample_weight=sample_weights_right)
        node.children.append(left_child)
        node.children.append(right_child)

        return node

    def fit(self, X: np.array, y: np.array, sample_weight=None):
        self.tree = self.build_tree(X, y, level=0, sample_weight=sample_weight)

    def predict(self, X):
        def _predict(x, node=None):
            if node is None:
                node = self.tree

            if node.split_feature is None:
                return node.classification

            if x[node.split_feature] <= node.split_value:
                return _predict(x, node=node.children[0])

            else:
                return _predict(x, node=node.children[1])

        y_predict = np.apply_along_axis(lambda x: _predict(x), 1, X)
        return y_predict
