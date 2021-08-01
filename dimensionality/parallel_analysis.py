from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent import futures
from itertools import repeat, starmap
from functools import partial

import numpy as np

from RyStats.common.polychoric import polychoric_correlation_serial
from RyStats.factoranalysis import principal_components_analysis as pca


__all__ = ["parallel_analysis"]


def _get_correlation_function(method):
    """Returns the correlation function. """
    if method[0] == 'pearsons':
        func = np.corrcoef
    elif method[0] == 'polychoric':
        func = partial(polychoric_correlation_serial, start_val=method[1], stop_val=method[2])
    else:
        raise ValueError('Unknown correlation method {}'.format(method[0]))

    return func


def parallel_analysis_serial(raw_data, n_iterations, correlation=('pearsons',), seed=None):
    """Estimate dimensionality from random data permutations.

    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data
        seed:  (integer) Random number generator seed value

    Returns
        eigs: mean eigenvalues
        sigma: standard deviation of eigenvalues
    """
    rng = np.random.default_rng(seed)

    # Get Seeds for repeatablity
    random_seeds = rng.choice(100*n_iterations, n_iterations, replace=False)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)

    correlation_method = _get_correlation_function(correlation)

    eigenvalue_array = np.zeros((n_iterations, n_items))

    for ndx, rseed in enumerate(random_seeds):
        rng_local = np.random.default_rng(rseed)

        new_data = rng_local.permutation(raw_data, axis=1).reshape(n_items, -1)
        local_correlation = correlation_method(new_data)

        _, eigenvalues, _ = pca(local_correlation)

        eigenvalue_array[ndx] = eigenvalues
    
    return eigenvalue_array.mean(0), eigenvalue_array.std(0, ddof=1)


def parallel_analysis(the_data, n_iterations=1000, method='pca',
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
