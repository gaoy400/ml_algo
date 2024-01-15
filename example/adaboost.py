from lib import ClassificationAdaboost
from data.chi_dist import chi_dist_data
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


def preprocess(n_sample, n_feature, test_size):
    X, y = chi_dist_data(n_sample, n_feature=n_feature)
    X = pd.DataFrame(X, columns=range(1, n_feature + 1))
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def main(in_sample=False):
    X_train, X_test, y_train, y_test = preprocess(3000, 10, 0.33)

    adaboost = ClassificationAdaboost(max_tree_level=1, min_loss_update=1e-6)

    t0 = time.time()
    adaboost.fit(X_train.values, y_train.values, max_iter=100)
    t1 = time.time()
    print(f'build tree costs {t1 - t0} seconds')

    if in_sample:
        X_test, y_test = X_train, y_train

    y_predict = adaboost.predict(X_test)

    diff = pd.Series(np.argmax(y_predict, axis=1) - y_test.values, index=y_test.index)
    accuracy = (diff == 0).sum() / len(y_test)
    print(f'Predict accuracy: {accuracy}')


if __name__ == '__main__':
    main(in_sample=True)


