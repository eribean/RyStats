import numpy as np
from scipy.optimize import minimize

from RyStats.factoranalysis import principal_components_analysis as pca


def _minres_min_func(unique_variance, correlation_matrix, n_factors, 
                     correlation_diagonal, indices):
    """Min function for minimum residual factor analysis"""
    np.fill_diagonal(correlation_matrix, 
                     correlation_diagonal - unique_variance)

    eigs, vects = np.linalg.eigh(correlation_matrix)

    eigs = eigs[-n_factors:]
    vects = vects[:, -n_factors:]
    vects2 = vects * eigs.reshape(1, -1)
    
    updated_corr = vects @ vects2.T
    lower_difference = correlation_matrix[indices] - updated_corr[indices]
    
    return np.square(lower_difference).sum()


def minres_factor_analysis(input_matrix, n_factors, initial_guess=None):
    """Performs minimum residual factor analysis.

    Minimium Residual factor analysis is equivalent to unweighted least-squares
    and also equal to Principal Axis Factor if Reduced Matrix remains positive
    definite.

    Args:
        input_matrx: Correlation or Covariance Matrix
        n_factors:  number of factors to extract
        initial_guess: Guess to seed the search algorithm

    Returns:
        loadings: unrotated extracted factor loadings
        eigenvalues: eigenvalues of extracted factor loadings
        unique_variance: variance unique to each item
    """
    working_matrix = input_matrix.copy()
    diagonal_from_input = np.diag(input_matrix)
    lower_indices = np.tril_indices_from(input_matrix, k=-1)
    
    # Initial Guess
    loads, _, _ = pca(input_matrix, n_factors)
    
    if initial_guess is None:
        uvars = np.diag(input_matrix - loads @ loads.T).copy()
    else:
        uvars = initial_guess.copy()

    args = (working_matrix, n_factors, diagonal_from_input, lower_indices)
    bounds = [(0.01, .99 * upper) for upper in diagonal_from_input]

    result = minimize(_minres_min_func, uvars, args, method='SLSQP',
                      bounds=bounds)
    unique_variance = result['x']

    loads, eigs, _ = pca(input_matrix - np.diag(unique_variance), n_factors)

    return loads, eigs, unique_variance