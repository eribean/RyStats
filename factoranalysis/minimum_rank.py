import copy

import numpy as np
from numpy.linalg import svd
from scipy.optimize import minimize
from scipy.linalg import eigvalsh, eigvals

from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import principal_axis_factor as paf
from RyStats.common import hyperspherical_angles, hyperspherical_vector


ZERO_TOL = -1e-6


__all__ = ['minimum_rank_factor_analysis']


def _spd_unique_variance(correlation_matrix, unique_variance):
    """Forces the smallest eigenvalue in the variance to zero.

    This results in the equation:

        adjusted_matrix = correlation_matrix - adjusted_variance

    being semi-positive definite.  The unique variance is scaled until the
    smallest eigenvalue is zero

    Args:
        correlation_matrix:  Input Correlation matrix
        unique_variance:  Input Variance Guess

    Returns:
        adjusted variance
    """
    epsilon = 0.02

    # Best Case, resulting matrix remains positive definite
    if unique_variance.min() > epsilon:
        half_inverse = 1 / np.sqrt(unique_variance)
        adjusted_matrix = correlation_matrix * np.outer(half_inverse, half_inverse)
        alpha = eigvalsh(adjusted_matrix, subset_by_index=[0, 0])

    else:
        adjusted_matrix = np.diag(unique_variance) @ np.linalg.inv(correlation_matrix)
        alpha = 1. / np.real(eigvals(adjusted_matrix)).max()

    return alpha * unique_variance


def _mrfa_min_func(unique_angles, correlation_matrix, n_factors):
    """Min function for minimum rank factor analysis"""
    # Convert the angles into cartesian space
    unique_variance = hyperspherical_vector(unique_angles)
    adjusted_variance = _spd_unique_variance(correlation_matrix, unique_variance)

    capital_psi = np.diag(adjusted_variance)
    reduced_matrix = correlation_matrix - capital_psi
    eigs = svd(reduced_matrix, compute_uv=False, hermitian=True)

    cost = np.inf
    if eigs[-1] >= ZERO_TOL:
        cost = eigs[:n_factors].sum()

    return cost
    

def minimum_rank_factor_analysis(correlation_matrix, n_factors,
                                 initial_guess=None,):
    """Performs minimum rank factor analysis on a correlation matrix.

    This method constrains the search region to force the resulting matrix to
    be semi-positive definite.

    Args:
        correlation_matrix:  input array to decompose (m x m)
        n_factors:  number of factors to keep
        initial_guess: Guess to seed the search algorithm, defaults to
                       the result of principal axis factor

    Returns:
        loadings: extracted factor loadings
        eigenvalues: extracted eigenvalues
        unique_variance: estimated unique variance
    """
    if initial_guess is None:
        _, _, initial_guess = paf(correlation_matrix, n_factors)

    args = (correlation_matrix, n_factors)
    bounds = [(0.01, 0.99*np.pi/2),] * (correlation_matrix.shape[0] - 1)
    initial_guess = hyperspherical_angles(initial_guess)

    result = minimize(_mrfa_min_func, initial_guess, args, method='SLSQP',
                      bounds=bounds)
    
    # Convert angles into cartesian space and calculate the outputs
    unique_variance = hyperspherical_vector(result['x'])
    adjusted_variance = _spd_unique_variance(correlation_matrix, unique_variance)

    return (pca(correlation_matrix - np.diag(adjusted_variance), n_factors)[:2] +
            (adjusted_variance,))
