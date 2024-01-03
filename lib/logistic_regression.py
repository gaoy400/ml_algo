"""
Logistic regression (binary classes)

Created on 10/4/2020

@author: Yuan Gao (gaoy400@gmail.com)
"""

import numpy as np


def initialization(dim):
    return np.ones(dim)


def logistic_gradient(data, label, theta):
    exp_transformed_data = np.exp(-data @ theta.reshape((-1, 1)))
    gradient_ = (label.astype(np.int32) - 1).reshape((-1, 1)) * data + exp_transformed_data * data / (1 + exp_transformed_data)
    return np.sum(gradient_, axis=0)


def logistic_likelihood(data, label, theta):
    transformed_data = data @ theta.reshape((-1, 1))
    likelihood_vector = (label.astype(np.int32) - 1).reshape((-1, 1)) * transformed_data - np.log((1 + np.exp(-transformed_data)))
    return np.sum(likelihood_vector, axis=0)[0]


def logit_reg_fit(data, label, learning_rate, decay=0.999):
    m, dim = data.shape
    likelihood_pre = -100
    theta = initialization(dim)

    while True:
        gradient = logistic_gradient(data, label, theta)
        theta += learning_rate * (decay ** 1) * gradient
        likelihood = logistic_likelihood(data, label, theta)

        if abs(likelihood - likelihood_pre) < 0.1:
            break

        likelihood_pre = likelihood

    return theta


def logit_reg_run(data, theta):
    posterior_1 = 1 / (1 + np.exp(-data @ theta.reshape(-1, 1)))
    posterior_0 = 1 - posterior_1
    label = np.argmax(np.concatenate((posterior_0, posterior_1)))
    return label
