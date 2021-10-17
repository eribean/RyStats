import numpy as np


__all__ = ["mcdonald_omega"]


def mcdonald_omega(loadings, uniqueness):
    """Computes McDonalds Omega for internal reliability.

    This metric accounts for differences in loadings for a 
    UNI-dimensional analysis. Items must be reverse scored 
    so that all loadings are positive.

    Args:
        loadings: [items x 1] loadings on a one-factor construct
        uniqueness: [items x 1] unique variance on a one-factor construct

    Returns:
        omega_t: McDonalds omega for internal consistency

    Notes:
        This is run after a ONE-FACTOR factor analysis using Maximum Likelihood
    """
    loadings = loadings.squeeze()
    uniqueness = uniqueness.squeeze()

    if np.ndim(loadings) > 1 or np.ndim(uniqueness) > 1:
        raise AssertionError("Inputs must be one dimensional vectors")

    numerator = np.square(loadings.sum())   
    denominator = uniqueness.sum()

    return numerator / (numerator + denominator)
