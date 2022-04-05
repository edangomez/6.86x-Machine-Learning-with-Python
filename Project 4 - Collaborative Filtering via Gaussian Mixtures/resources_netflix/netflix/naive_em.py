"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, pi = mixture

    N = (2 * np.pi * var) ** (-d / 2) * np.exp(-np.linalg.norm(X[:, np.newaxis] - mu, axis=2) ** 2 / (2 * var))
    P = pi * N
    p_x_thet = P.sum(axis=1).reshape(n, 1)
    soft_counts = (pi * N) / p_x_thet
    logL = np.log(p_x_thet)

    return soft_counts, np.sum(logL, axis=0)

    # raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    n_ = np.sum(post, axis =0)
    p_ = n_/n
    mu_ = (1/n_.reshape(K,1))*(post.T @ X)
    norm = np.power(np.linalg.norm(X[:, np.newaxis] - mu_, axis = 2), 2)
    var_ = (1/(n_*d))*np.sum(post*norm, axis = 0)

    # n_ = np.sum(post, axis=0)  # Get the N_hat by adding up the posterior probability by column
    # p_ = n_ / n
    # mu_ = (1 / n_.reshape(K, 1)) * post.T @ X
    # norm = np.power(np.linalg.norm(X[:, np.newaxis] - mu_, axis=2), 2)
    # var_ = (1 / (n_ * d)) * np.sum(post * norm, axis=0)
    return GaussianMixture(mu_, var_, p_)

    return GaussianMixture(mu_, var_, p_)

    # raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # raise NotImplementedError

    L_old = None
    L_new = None
    epsilon = 1e-6
    while L_old is None or abs(L_old - L_new) > epsilon*abs(L_new):
        L_old = L_new
        p, L_new = estep(X,mixture)
        mixture = mstep(X, p)

    return mixture, p, L_new