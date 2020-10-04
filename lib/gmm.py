"""
Implementation of Gaussian Mixture Model using EM algorithm.

Created on 10/4/2020

@author: Yuan Gao (gaoy400@gmail.com)
"""

import numpy as np


def gaussian_dist_pdf(x, mu, sigma):
    # ignore (pi*2)^(dim/2) factor to enhance numerical stability
    m, dim = x.shape
    prob = np.exp(np.diag(-0.5 * (x - mu).dot(np.linalg.inv(sigma)).dot((x - mu).T))) / np.sqrt(np.linalg.det(sigma))
    return prob / (2*np.pi) ** (dim / 2.)


def initialization(k, dim):
    np.random.seed(0)
    pi = np.full((k, ), 1. / k)
    mu = np.random.normal(size=k*dim).reshape((k, -1))
    sigma = np.random.normal(size=k*dim*dim).reshape((k, dim, dim))
    sigma[0] = sigma[0].dot(sigma[0].T) + np.identity(dim)
    sigma[1] = sigma[1].dot(sigma[1].T) + np.identity(dim)
    return pi, mu, sigma


def step_expectation(data, pi, mu, sigma):
    k = pi.shape[0]
    joint_prob = []
    for i in range(k):
        joint_prob.append(pi[i] * gaussian_dist_pdf(data, mu[i], sigma[i]))
    joint_prob = np.array(joint_prob)
    x_marginal = np.sum(joint_prob, axis=0)
    tau = joint_prob / x_marginal
    return tau


def step_maximization(data, tau, mu):
    k, m = tau.shape
    _, dim = mu.shape
    pi_ = np.sum(tau, axis=1) / m
    mu_ = tau.dot(data) / np.sum(tau, axis=1).reshape((k, 1))

    sigma_ = []
    for i in range(k):
        tau_k, mu_k = tau[i].reshape((m, 1)), mu_[i].reshape((dim, 1))
        weighted_central_x = (data * np.sqrt(tau_k)).T - (mu_k * np.sqrt(tau_k).T)
        sigma_update = weighted_central_x.dot(weighted_central_x.T) / np.sum(tau_k)
        sigma_.append(sigma_update)
    sigma_ = np.array(sigma_)
    return pi_, mu_, sigma_


def gmm_likelihood(data, pi, mu, sigma):
    k = pi.shape[0]
    prob_x = []
    for i in range(k):
        prob_x.append(pi[i] * gaussian_dist_pdf(data, mu=mu[i], sigma=sigma[i]))
    prob_x = np.array(prob_x)
    prob_x = np.sum(prob_x, axis=0)
    log_likelihood = np.sum(np.log(prob_x))
    return log_likelihood


def gmm_em(data, k=2):
    """

    :param data: shape = m*dim, represents the preprocessed data set for clustering
    :param k: components amount
    :return: a label array specifies the cluster number of each record of data
    """
    m, dim = data.shape
    pi, mu, sigma = initialization(k, dim)
    likelihood_0 = -100
    for i in range(100):
        tau = step_expectation(data, pi, mu, sigma)
        pi, mu, sigma = step_maximization(data, tau, mu)
        likelihood = gmm_likelihood(data, pi, mu, sigma)
        if abs(likelihood - likelihood_0) < 0.1:
            break
        likelihood_0 = likelihood

    label = np.argmax(tau, axis=0)
    return label
