#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Test Gaussian Mixture Model on MNIST data set.

Created on 10/4/2020

@author: Yuan Gao (gaoy400@gmail.com)
"""
import os

import numpy as np
import scipy.io as spio
from sklearn.decomposition import PCA

from lib import gmm_em

dir_name = os.path.dirname(__file__)
data_file_name = os.path.join(dir_name, '../data/mnist_data.mat')
label_file_name = os.path.join(dir_name, '../data/mnist_label.mat')


def data_preprocess(pca_dim=5):
    image_gallery = spio.loadmat(data_file_name)['data'].T
    true_label = spio.loadmat(label_file_name)['trueLabel'][0]

    pca = PCA(n_components=pca_dim)
    pca.fit(image_gallery)
    image_data = pca.transform(image_gallery)

    return image_data, true_label


def test_gmm():
    # The data set only includes all the "2" and "6" images.
    image_data, true_label = data_preprocess()
    label = gmm_em(data=image_data, k=2)

    true_label[true_label == 2] = 1
    true_label[true_label == 6] = 0

    mismatch_rate = min(np.mean(label == true_label), np.mean(label != true_label))
    return mismatch_rate


if __name__ == '__main__':
    mismatch_rate = test_gmm()
    print(mismatch_rate)

