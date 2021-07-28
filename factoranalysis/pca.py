import numpy as np


__all__ = ['principal_components_analysis']


def principal_components_analysis(input_matrix, n_factors):
    """Performs principal components analysis.

    Args:
        input_matrix: input correlation or covariance matrix
        n_factors: number of factors to extract

    Returns:
        loadings: factor loadings
        eigenvalues: eigenvalues for the factors
        unique_variance: vector of all zeros
    """
    if n_factors is None:
        n_factors = input_matrix.shape(0)

    # Symmetric Matrices have an svd equal to np.linalg.eigh
    uu, sv, _ = np.linalg.svd(input_matrix, hermitian=True)

    # Only return the requested factors
    eigenvalues = sv[:n_factors]
    loadings = uu[:, :n_factors]

    # Scale Loadings
    loadings /= np.sqrt(eigenvalues)

    return loadings, eigenvalues, np.zeros((input_matrix.shape(0)))