from lib import ClassificationDecisionTree
import numpy as np


class ClassificationAdaboost:

    def __init__(self, max_tree_level=1, min_loss_update=0, impurity_method="gini"):
        self.max_tree_level = max_tree_level
        self.min_loss_update = min_loss_update
        self.impurity_method = impurity_method

    def fit(self, X, y, max_iter=500):
        """Implementation of Hastie, Trevor, et al. 'Multi-class adaboost.'"""
        n_class = len(set(y))

        n_sample, n_feature = X.shape

        Y = -np.ones((n_sample, n_class), dtype=float) / (n_class - 1)
        Y[np.arange(n_sample), y] = 1

        w = np.ones((n_sample,))
        cdt_list = []
        err_list = []
        beta_list = []
        loss_list = []

        for i in range(max_iter):
            print(f'iteration: {i}')
            cdt = ClassificationDecisionTree(
                max_level=self.max_tree_level,
                impurity_method=self.impurity_method
            )
            cdt.fit(X, y, sample_weight=w / sum(w))
            y_pred = cdt.predict(X)
            err = ((y_pred != y) @ w) / sum(w)
            beta = (n_class - 1) ** 2 / n_class * (np.log((1 - err) / err) + np.log(n_class - 1))

            G = -np.ones((n_sample, n_class), dtype=float) / (n_class - 1)
            G[np.arange(n_sample), y_pred] = 1

            w = w * np.exp(-beta * np.sum(Y * G, 1) / n_class)
            loss = np.sum(w)

            if (
                len(loss_list) > 0
                and abs(loss / loss_list[-1] - 1) < self.min_loss_update
            ):
                print('break')
                break

            cdt_list.append(cdt)
            err_list.append(err)
            beta_list.append(beta)
            loss_list.append(loss)
            print('err: ', err, 'loss: ', loss)

        self.cdt_list = cdt_list
        self.err_list = err_list
        self.beta_list = beta_list
        self.loss_list = loss_list
        self.classes = set(y)

    def predict(self, X):
        n_class = len(self.classes)
        n_sample, n_feature = X.shape

        f = np.zeros((n_sample, n_class))
        for (cdt, err, beta) in zip(self.cdt_list, self.err_list, self.beta_list):
            y_pred = cdt.predict(X)
            g = -np.ones((n_sample, n_class), dtype=float) / (n_class - 1)
            g[np.arange(n_sample), y_pred] = 1
            f += beta * g

        prob = np.exp(1 / (n_class - 1) * f)
        prob = prob / np.sum(prob, axis=1).reshape((-1, 1))
        return prob

