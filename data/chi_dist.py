import numpy as np
from scipy.stats import chi2


def chi_dist_data(n_sample, n_feature):
    np.random.seed(0)
    X = np.random.normal(size=(n_sample, n_feature))
    X_ = np.sum(X * X, axis=1)

    split = chi2.ppf([1 / 3., 2 / 3.], df=n_feature)

    y = np.zeros((n_sample, ))
    y[(X_ < split[1]) & (X_ >= split[0])] = 1
    y[X_ >= split[1]] = 2

    return X, y.astype(int)
