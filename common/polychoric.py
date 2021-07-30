from concurrent import futures
from itertools import product, repeat
from functools import reduce
from operator import add

import numpy as np
from numba import njit
from scipy.optimize import optimize
from scipy.special import erfinv
from scipy.stats import mvn


def validate_contigency_table(the_table):
    """Checks tables for columns or rows with all zeros and deletes.

    Args:
        the_table: input contingency table

    Returns:
        updated table with columns/rows corresponding to all zeros
    """
    mask = list()
    for ndx in range(2):
        sums = the_table.sum(axis=ndx)
        mask.append(sums != 0)
    return the_table[mask[1]][:, mask[0]]


def contingency_table(vals1, vals2, start_val=1, stop_val=5):
    """Creates a contingency table for two ordinal values.

    Args:
        vals1: Input vector of ordinal data
        vals2: Input Vector of ordinal_data
        start_val: first value of record
        stop_val:  last value of record

    Returns:
        array of counts for each pair in (vals1, vals2)
    """
    n_vals = stop_val - start_val + 1
    cont_table = np.zeros((n_vals, n_vals))

    linear_ndx = vals1 * (stop_val+1) + vals2
    for ndx1, ndx2 in product(range(1, n_vals+1), range(1, n_vals+1)):
        cont_table[ndx1-1, ndx2-1] = _local_count(linear_ndx,
                                                  ndx1*(stop_val+1) + ndx2)
    return cont_table


def bivariate_normal_integral(S, low, upp, mu= [0., 0.]):
    """Computes the integeral for a bivariate normal distribution.

    Args:
        SS: Correlation matrix
        low: (l1, l2) tuple of lower bounds of integration
        upp: (u1, u2) tuple of upper bounds of integration
        mu:  (m1, m2) tuple of means in each direction

    Returns:
        Result of integration representing the probability in section
    """
    return mvn.mvnun(low, upp, mu, S, abseps=1e-12)[0]


def polychoric_correlation(vals1, vals2, start_val=1, stop_val=5):
    """Computes the polychoric correlation coefficient.

    Args:
        vals1:  Vector of data
        vals2:  Vector of data
        start_val: starting value of ordinal in vals1 and vals2
        stop_val: stopping value of ordinal in vals1 and vals2

    Returns:
        polychoric correlation coefficient
    """
    the_table = contingency_table(vals1, vals2, start_val, stop_val)
    the_table = validate_contigency_table(the_table)
    the_probabilities = np.zeros_like(the_table)

    # Determine the threshold values in each direction
    threshs = list()
    for ndx in range(2):
        norm_vals = np.cumsum(the_table.sum(axis=ndx))
        norm_vals /= norm_vals[-1]
        thresh = np.sqrt(2) * erfinv(2 * norm_vals[:-1] - 1)
        threshs.append(np.concatenate(([-23.0], thresh, [23.0])))

    corr = np.eye(2)
    mu = np.array([0, 0])
    # Minimization function, assumes (x, y) not (row, column)
    def min_func(rho):
        corr[0, 1] = rho
        corr[1, 0] = rho
        for ndx1, ndx2 in product(range(threshs[1].size-1), range(threshs[0].size-1)):
            low = (threshs[1][ndx1], threshs[0][ndx2])
            upp = (threshs[1][ndx1+1], threshs[0][ndx2+1])
            the_probabilities[ndx1, ndx2] = mvn.mvnun(low, upp, mu, corr, abseps=1e-12)[0]
        return -1 * np.sum(the_table * np.log(the_probabilities.clip(1e-46, None)))

    return optimize.fminbound(min_func, -0.999, .999)


def polychoric_matrix(ordinal_data, start_val=1, stop_val=5, subset_range=None):
    """Creates correlation matrix of polychoric correlations.

    Analagous to a correlation coefficient matrix, except polychloric
    correlations are used in lieu of pearsons coefficient.

    Args:
        ordinal_data: matrix of ordinal vectors where
                      ordinal_data.shape[0] is the number of items
                      ordinal_data.shape[1] is the number of observations in each vector
        start_val: starting value of ordinal data
        stop_val: stopping value of ordinal data
        subset_range: iterator of column indices to compute elements for

    Returns:
        Square matrix of shape (ordinal_data.shape[0], ordinal_data.shape[0])
        with 1's on the diagonal and correlations on the off-diagonals
    """
    if subset_range is None:
        subset_range = range(1, ordinal_data.shape[0])

    corr_matrix = np.eye(ordinal_data.shape[0])

    for row_ndx in subset_range:
        for col_ndx in range(0, row_ndx):
            corr_matrix[row_ndx, col_ndx] = polychoric_correlation(ordinal_data[row_ndx],
                                                                   ordinal_data[col_ndx],
                                                                   start_val, stop_val)
            corr_matrix[col_ndx, row_ndx] = corr_matrix[row_ndx, col_ndx]

    return corr_matrix


def polychoric_matrix_parallel(ordinal_data, start_val=1, stop_val=5, num_processors=2):
    """Uses multiprocessing to compute the polychoric matrix.

    Args:
        ordinal_data: matrix of ordinal vectors where
                      ordinal_data.shape[0] is the number of items
                      ordinal_data.shape[1] is the number of observations in each vector
        start_val: starting value of ordinal data
        stop_val: stopping value of ordinal data
        num_processors: number of multiprocessors to use

    Returns:
        Square matrix of shape (ordinal_data.shape[0], ordinal_data.shape[0])
        with 1's on the diagonal and correlations on the off-diagonals
    """
    if num_processors == 1:
        return polychoric_matrix(ordinal_data, start_val, stop_val)

    matrix_shape = ordinal_data.shape[0]
    subsets = [range(ndx, matrix_shape, num_processors)
               for ndx in range(1, num_processors+1)]

    with futures.ProcessPoolExecutor(max_workers=num_processors) as pool:
        results = pool.map(polychoric_matrix, repeat(ordinal_data),
                           repeat(start_val), repeat(stop_val), subsets)
        corr_matrix = reduce(add, results)

    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


def non_numba_count(array, value):
    """Function to count number of items in array w/o numba, slower."""
    return np.count_nonzero(array == value)


@njit
def _local_count(array, value):
    """Function to fast count the number of items in array."""
    the_sum = 0
    for array_value in array:
        if array_value == value:
            the_sum += 1
    return the_sum