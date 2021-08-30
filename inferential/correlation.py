import numpy as np

from scipy import stats as sp


__all__ =  ['pearsons_correlation']


def pearsons_correlation(raw_data):
    """Computes the correlation and statistics for a dataset.

    Args:
        raw_data:  Data matrix [n_items, n_observations]

    Returns:
        dict: Dictionary of correlation, and critical rho values

    Notes:
        The integration is over the n_observations such that the output is
        of size [n_items, n_items]
    """
    correlation = np.corrcoef(raw_data)

    # Compute the critical values for the 3 significance tests
    deg_of_freedom = raw_data.shape[1] - 2
    t_critical = sp.t.isf([.025, .005, 0.0005] , deg_of_freedom)
    r_critical = np.sqrt(t_critical**2 / (t_critical**2 + deg_of_freedom))

    return {'Correlation': correlation, 
            'R critical': {'.05': r_critical[0],
                           '.01': r_critical[1],
                           '.001': r_critical[2]},
           }   
