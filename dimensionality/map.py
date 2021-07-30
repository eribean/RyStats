import numpy as np


def _get_correlation_method(correlation):
    if correlation[0].lower() == 'pearsons':
        func = np.corrcoef
    elif correlation[0].lower() == 'polychoric':
        func = partial(polychoric_matrix, start_val=correlation[1], stop_val=correlation[2])
    else:
        raise ValueError('Unknown correlation method {}'.format(correlation[0]))
        
    return func


def dimensionality_map(the_data, correlation=('pearsons', )):
    """ Determines dimensionality based on the matrix of partial correlations.

    Args:
        the_data:  Input array with the_data.shape == (n_items, n_observations)
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data

    Returns
        ratio of partial correlation squared-sum to correlation squared-sum
    """    
    func = _get_correlation_method(correlation)
    
    correlation_matrix = func(the_data)
    diag_inds = np.tril_indices_from(correlation_matrix, k=-1)
    denominator = np.square(correlation_matrix[diag_inds]).sum()
    
    # Principal Components
    the_ratio = list()
    for n_factors in range(1, the_data.shape[0] - 1):
        facts, _, _ = pca(correlation_matrix, n_factors=n_factors)

        partial_difference = correlation_matrix - facts @ facts.T
        weighting = 1.0 / np.sqrt(np.diag(partial_difference))

        partial_correlation = partial_difference * np.outer(weighting, weighting)
        the_ratio.append(np.square(partial_correlation[diag_inds]).sum() / denominator)

    return np.argmin(the_ratio) + 1, the_ratio