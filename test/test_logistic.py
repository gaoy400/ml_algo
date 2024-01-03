#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Test logistic regression classifier on MNIST data set.

Created on 10/4/2020

@author: Yuan Gao (gaoy400@gmail.com)
"""

from sklearn.decomposition import PCA

from data.mnist import get_mnist_data_label
from lib import logit_reg_fit, logit_reg_run
from test.utils import get_mismatch_rate


def preprocess(pca_dim=2, test_portion=0.8):
    image_gallery, true_label = get_mnist_data_label()

    true_label[true_label == 2] = 0
    true_label[true_label == 6] = 1

    m, _ = image_gallery.shape
    train_set_amount = int(m * test_portion)

    pca = PCA(n_components=pca_dim)
    pca.fit(image_gallery[:train_set_amount, :])
    image_data_train = pca.transform(image_gallery[:train_set_amount, :])
    true_label_train = true_label[:train_set_amount]

    pca = PCA(n_components=pca_dim)
    pca.fit(image_gallery[train_set_amount:, :])
    image_data_test = pca.transform(image_gallery[train_set_amount:, :])
    true_label_test = true_label[train_set_amount:]

    return image_data_train, true_label_train, image_data_test, true_label_test


def run_logistic_regression():
    image_data_train, true_label_train, image_data_test, true_label_test = preprocess()
    theta = logit_reg_fit(data=image_data_train, label=true_label_train, learning_rate=0.01)
    label = logit_reg_run(data=image_data_test, theta=theta)

    mismatch_rate = get_mismatch_rate(label=label, true_label=true_label_test)
    return mismatch_rate


if __name__ == '__main__':
    mismatch_rate = run_logistic_regression()
    print('mismatch_rate: ', mismatch_rate)