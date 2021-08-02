from functools import partial
import numpy as np


__all__ = ["minimum_average_partial"]


def minimum_average_partial(correlation_matrix):
    """ Determines dimensionality based on partial correlations.

    Args:
        correlation_matrix:  Input correlation matrix from pearsons 
                             or polychoric correlations

    Returns
        n_factors: number of factors determined
        metric: squared average partial correlations
        
    Note:
        Based on Velicer's MAP test, https://doi.org/10.1007/BF02293557
    """    
    n_items = correlation_matrix.shape[0]
    scalar = 2 / (n_items * (n_items - 1))
    
    diag_inds = np.tril_indices_from(correlation_matrix, k=-1)
    
    the_ratio = np.zeros((n_items-1))
    the_ratio[0] = scalar * np.square(correlation_matrix[diag_inds]).sum()

    # Basic Principal Components
    u1, s1, _ = np.linalg.svd(correlation_matrix, hermitian=True)
    s1 = np.sqrt(s1)

    for ndx, n_factors in enumerate(range(1, n_items - 1)):
        facts = u1[:, :n_factors] * s1[:n_factors]

        partial_difference = correlation_matrix - facts @ facts.T
        weighting = 1.0 / np.sqrt(np.diag(partial_difference))

        partial_correlation = partial_difference * np.outer(weighting, weighting)
        the_ratio[ndx+1] = scalar * np.square(partial_correlation[diag_inds]).sum() 

    return np.argmin(the_ratio), the_ratio