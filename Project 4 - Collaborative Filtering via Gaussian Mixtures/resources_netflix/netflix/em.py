"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # raise NotImplementedError
    # mu, var, p = mixture
    # n = X.shape[0]
    #
    # sparese_matrix = np.whera(X>0, 1, 0)
    # d = np.sum(sparese_matrix, axis= 1).reshape((n, 1))
    # A = -d/2 * np.log(2 * np.pi *var)
    # B = np.linalg.norm((X[:, np.newaxis] - mu) * sparse_matrix[:, np.newaxis], axis=2) ** 2 / -(2 * var)
    # log_N = A + B
    # log_pi =  np.log(p + 1e-16)
    # f_uj = log_pi + log_N
    #
    #
    # log_p = f_uj - logsumexp(f_uj, axis = 1).reshape((n, 1))
    # Log_L = np.sum(logsumexp(f_uj, axis = 1).reshape((n, 1)), axis = 0)
    # soft_count = np.exp(log_p)
    #
    #
    # return Log_L, soft_count

    n = X.shape[0]
    mu, var, p = mixture
    sparse_matrix = np.where(X > 0, 1, 0)
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))

    A = -d / 2 * np.log(2 * np.pi * var)
    B = np.linalg.norm((X[:, np.newaxis] - mu) * sparse_matrix[:, np.newaxis], axis=2) ** 2 / -(2 * var)
    log_pi = np.log(p + 1e-16)
    log_N = A + B
    f_uj = log_pi + log_N
    log_p = f_uj - logsumexp(f_uj, axis=1).reshape((n, 1))

    soft_count = np.exp(log_p)
    log_L = np.sum(logsumexp(f_uj, axis=1).reshape((n, 1)), axis=0)

    return soft_count, log_L


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # raise NotImplementedError
    # n = X.shape[0]
    # mu, var, p = mixture
    # sparse_matrix = np.where(X > 0, 1, 0)
    # d = np.sum(sparse_matrix, axis=1).reshape((n, 1))
    #
    # n_ = np.sum(post, axis = 0)
    # Log_p_ = np.log(n_) - np.log(n)
    # p_ = np.exp(Log_p_)
    # mu_d = post.T @ sparse_matrix

    mu, var, p = mixture
    n = X.shape[0]
    sparse_matrix = np.where(X > 0, 1, 0)
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))

    # n estimator
    n_ = np.sum(post, axis=0)
    # p estimator
    log_p_ = np.log(n_) - np.log(n)
    p_ = np.exp(log_p_)

    mu_d = post.T @ sparse_matrix
    mu_update = np.where(mu_d >= 1, 1, 0)
    div = np.divide(post.T @ X, mu_d, out=np.zeros_like(post.T @ X), where=mu_d != 0)
    mu_hat = np.where(div * mu_update == 0, mu, div)

    var_d = np.sum(d * post, axis=0)
    norm_Cu = np.linalg.norm((X[:, np.newaxis] - mu_hat) * sparse_matrix[:, np.newaxis], axis=2)**2
    var_ = np.divide(np.sum(post * norm_Cu, axis=0), var_d, out=np.zeros_like(np.sum(post * norm_Cu, axis=0)), where = var_d != 0)
    var_withmin = np.where(var_ > min_variance, var_, min_variance)


    return GaussianMixture(mu_hat, var_withmin, p_)







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
    L_old = None
    L_new = None
    epsilon = 1e-6
    while L_old is None or abs(L_old - L_new) > epsilon * abs(L_new):
        L_old = L_new
        p, L_new = estep(X, mixture)
        mixture = mstep(X, p, mixture)

    return mixture, p, L_new


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_copy = X.copy()
    # mu, var, p = mixture

    # estep

    n = X.shape[0]
    mu, var, p = mixture
    sparse_matrix = np.where(X > 0, 1, 0)
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))

    A = -d / 2 * np.log(2 * np.pi * var)
    B = np.linalg.norm((X_copy[:, np.newaxis] - mu) * sparse_matrix[:, np.newaxis], axis=2) ** 2 / -(2 * var)
    log_pi = np.log(p + 1e-16)
    log_N = A + B
    f_uj = log_pi + log_N
    log_p = f_uj - logsumexp(f_uj, axis=1).reshape((n, 1))

    soft_count = np.exp(log_p)
    log_L = np.sum(logsumexp(f_uj, axis=1).reshape((n, 1)), axis=0)

    post = soft_count

    indices = np.where(X_copy != 0, 1, 0)
    predicted_value = post @ mu
    X_pred = np.where(indices * X_copy == 0, predicted_value, X_copy)

    return X_pred
