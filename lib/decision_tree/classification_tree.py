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
    ):
        self.max_level = max_level
        self.tree = None

    @classmethod
    def __sample_prob(cls, y):
        n = len(y)
        _, counts = np.unique(y, return_counts=True)
        p = counts / n
        return p

    @classmethod
    def gini_index(cls, y):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y)
        gini = sum(p * (1 - p))
        return gini

    @classmethod
    def cross_entropy(cls, y):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y)
        deviance = - sum(p * np.log(p))
        return deviance

    @classmethod
    def misclassification(cls, y):
        if len(y) == 0:
            return 0
        p = cls.__sample_prob(y)
        return 1 - np.max(p)

    def build_tree(self, X: np.array, y: np.array, impurity_method="gini", level: int = 0):

        if impurity_method == 'gini':
            impurity_func = self.gini_index
        elif impurity_method == 'entropy':
            impurity_func = self.cross_entropy
        elif impurity_method == 'misclassification':
            impurity_func = self.misclassification
        else:
            raise NotImplementedError(f'Impurity method {impurity_method} is not implemented.')

        classification = mode(y)
        current_node_impurity = impurity_func(y)
        if (self.max_level is not None and level >= self.max_level) or len(np.unique(y)) <= 1:
            return Node(
                split_feature=None,
                split_value=None,
                level=level,
                classification=classification,
                node_impurity=current_node_impurity,
            )

        feature_impurity = dict()
        feature_val = dict()

        for feature in range(X.shape[1]):

            min_node_impurity, split_val = 100 * X.shape[0], None

            for val in set(X[:, feature]):
                left_child_mask = X[:, feature] <= val
                right_child_mask = (~left_child_mask)

                y_left, y_right = y[left_child_mask], y[right_child_mask]

                node_impurity = (len(y_left) * impurity_func(y_left) + len(y_right) * impurity_func(y_right)) / len(y)

                if node_impurity < min_node_impurity:
                    min_node_impurity = node_impurity
                    split_val = val

            feature_impurity[feature] = min_node_impurity
            feature_val[feature] = split_val

        split_feature = min(feature_impurity, key=feature_impurity.get)
        split_val, node_impurity = feature_val[split_feature], feature_impurity[split_feature]

        left_child_mask = X[:, split_feature] <= split_val
        right_child_mask = (~left_child_mask)

        y_left, y_right = y[left_child_mask], y[right_child_mask]
        X_left, X_right = X[left_child_mask], X[right_child_mask]

        if len(y_left) > 0 and len(y_right) > 0:
            node = Node(
                split_feature=split_feature,
                split_value=split_val,
                level=level,
                classification=classification,
                node_impurity=node_impurity,
            )

            left_child = self.build_tree(
                X_left, y_left,
                impurity_method=impurity_method, level=level + 1
            )
            right_child = self.build_tree(
                X_right, y_right,
                impurity_method=impurity_method, level=level + 1
            )
            node.children.append(left_child)
            node.children.append(right_child)

        else:
            node = Node(
                split_feature=None,
                split_value=None,
                level=level,
                classification=classification,
                node_impurity=node_impurity,
            )

        return node

    def fit(self, X: np.array, y: np.array, impurity_method="gini"):
        self.tree = self.build_tree(X, y, impurity_method, level=0)

    def predict(self, x, node=None):
        if node is None:
            node = self.tree

        if node.split_feature is None:
            return node.classification
        
        if x[node.split_feature] <= node.split_value:
            return self.predict(x, node=node.children[0])
        
        else:
            return self.predict(x, node=node.children[1])
