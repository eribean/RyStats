import numpy as np
from numpy.linalg import svd
from scipy.optimize import minimize

from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import principal_axis_factor as paf


__all__ = ['minimum_rank_factor_analysis']


def _mrfa_min_func(inverse_half_variance, correlation_cholesky, n_factors):
    """Min function for minimum rank factor analysis"""
    eigs = svd(inverse_half_variance 
               * correlation_cholesky, compute_uv=False)
    return (eigs / eigs[-1])[n_factors:].sum()
    

def minimum_rank_factor_analysis(correlation_matrix, n_factors,
                                 initial_guess=None, n_iter=500):
    """Performs minimum rank factor analysis on a correlation matrix.

    This method constrains the search region to force the resulting matrix to
    be semi-positive definite.

    Args:
        correlation_matrix:  input array to decompose (m x m)
        n_factors:  number of factors to keep
        initial_guess: Guess to seed the search algorithm, defaults to
                       the result of principal axis factor
        n_iter: Maximum number of iterations to run (Default: 500)

    Returns:
        loadings: extracted factor loadings
        eigenvalues: extracted eigenvalues
        unique_variance: estimated unique variance
    """
    if initial_guess is None:
        _, _, initial_guess = paf(correlation_matrix, n_factors)

    correlation_cholesky = np.linalg.cholesky(correlation_matrix).T.copy()
    args = (correlation_cholesky, n_factors)
    
    initial_guess = 1. / np.sqrt(initial_guess)
    bounds = [(1, 100),] * (correlation_matrix.shape[0])
    result = minimize(_mrfa_min_func, initial_guess, args, method='SLSQP',
                      bounds=bounds, options={'maxiter': n_iter})
    
    # Convert result into unique variance
    eigs = svd(result['x'] * correlation_cholesky, compute_uv=False)
    
    unique_variance = np.square(eigs[-1] / result['x'])
    loadings, eigs, _ = pca(correlation_matrix 
                            - np.diag(unique_variance), n_factors)

    return loadings, eigs, unique_variance
