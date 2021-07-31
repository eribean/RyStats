from concurrent import futures
from itertools import repeat, starmap
from functools import partial

import numpy as np

from RyStats.common.polychoric import polychoric_correlation
from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import principal_axis_factor as paf


__all__ = ["parallel_analysis_from_data"]


def _get_extraction_function(method):
    """Returns the appropriate parallel analysis method."""
    return {'pca': pca,
            'paf': partial(paf, n_factors=1, force_spd=False)}[method]


def _get_correlation_function(method):
    """Returns the correlation function. """
    if method[0] == 'pearsons':
        func = np.corrcoef
    elif method[0] == 'polychoric':
        func = partial(polychoric_matrix, start_val=method[1], stop_val=method[2])
    else:
        raise ValueError('Unknown correlation method {}'.format(method[0]))

    return func


def _parallel_engine(random_matrix, correlation_method, extraction_method):
    """Runs the parallel analysis."""
    corr_matrix = correlation_method(random_matrix)

    try:
        results = extraction_method(corr_matrix)
    except (RuntimeError, ValueError):
        return np.full(corr_matrix.shape[0], np.nan)

    unique_variance = results[-1]

    # PCA doesn't return a variance vector
    if len(results) == 2:
        unique_variance *= 0.0

    return np.linalg.eigvalsh(corr_matrix - np.diag(unique_variance))


def parallel_analysis(n_items, n_observations, n_iterations=1000,
                      method='pca', rseed=None, num_processors=1,
                      _return_raw=False):
    """Creates eigenvalues for factor retention thresholding.

    Args:
        n_items:  Number of items in data matrix
        n_observations:  Number of observations in data matrix
        n_iterations:  Number of iterations to run
        method: Method to extract factors {'pca', 'paf', 'pafspd', 'mrfa'}
        rseed:  (integer) Random number generator seed value
        num_processors:  Number of processors to use, default is 1

    Returns
        tuple of:
            eigs: median eigenvalues
            sigma: standard deviation of eigenvalues
            invalid: number of invalid entries found
    """
    if rseed is not None:
        np.random.seed(rseed)

    extraction_method = _get_extraction_function(method)
    correlation_method = np.corrcoef

    random_matrices = starmap(np.random.randn,
                              repeat((n_items, n_observations), n_iterations))

    args = (_parallel_engine, random_matrices, repeat(correlation_method),
            repeat(extraction_method))

    return _parallel_analysis_abstract(args, num_processors, n_iterations,
                                       method, _return_raw)


def parallel_analysis_from_data(the_data, n_iterations=1000, method='pca',
                                correlation=('pearsons',), rseed=None,
                                num_processors=1, _return_raw=False):
    """Creates eigenvalues for factor retention from data.

    The input data is permuted n_iteration times to obtain the diversity.

    Args:
        the_data:  Input array with the_data.shape == (n_items, n_observations)
        n_iterations:  Number of iterations to run
        method: Method to extract factors {'pca', 'paf', 'pafspd', 'mrfa'}
        rseed:  (integer) Random number generator seed value
        num_processors:  Number of processors to use, default is 1
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data
    Returns
        tuple of:
            eigs: median eigenvalues
            sigma: standard deviation of eigenvalues
            invalid: number of invalid entries found
    """
    if rseed is not None:
        np.random.seed(rseed)

    extraction_method = _get_extraction_function(method)
    correlation_method = _get_correlation_function(correlation)

    random_matrices = map(np.apply_along_axis, repeat(np.random.permutation),
                          repeat(1), repeat(the_data, n_iterations))

    args = (_parallel_engine, random_matrices, repeat(correlation_method),
            repeat(extraction_method))
    return _parallel_analysis_abstract(args, num_processors, n_iterations,
                                       method, _return_raw)


def _parallel_analysis_abstract(args, num_processors, n_iterations, method,
                                _return_raw):
    """Common Function to run parallel analysis data."""
    if num_processors == 1:
        eigs = map(*args)
    else:
        with futures.ProcessPoolExecutor(max_workers=num_processors) as pool:
            eigs = pool.map(*args,
                            chunksize=int(n_iterations / 2 / num_processors))

    eigen_list = list(eigs)

    # To support distributed computing
    if _return_raw:
        return eigen_list

    eigenvalues = np.array(eigen_list)
    mask = np.isnan(eigenvalues[:, 0])
    eigenvalues = eigenvalues[~mask, :]

    # The median is more robust to outliers, and represents the 50th percentile
    median_eigenvalues = np.median(eigenvalues, axis=0)
    std_eigenvalues = eigenvalues.std(axis=0)

    # Adjust for 1-factor methods
    if method != 'pca':
        last_column = eigenvalues[:, -1]
        std_guess = np.median(std_eigenvalues) + std_eigenvalues[1:-1].std() * 9
        median_value = np.median(last_column)
        upper = median_value + std_guess
        lower = median_value - std_guess

        mask = (last_column >= lower) & (last_column <= upper)
        std_eigenvalues[-1] = last_column[mask].std()

    return median_eigenvalues[::-1], std_eigenvalues[::-1], mask.sum()
