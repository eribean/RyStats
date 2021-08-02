from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent import futures

from itertools import repeat, starmap
from functools import partial

import numpy as np
from numpy.random import SeedSequence

from RyStats.common.polychoric import polychoric_correlation_serial


__all__ = ["parallel_analysis", "parallel_analysis_serial"]


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
    # Get Seeds for repeatablity
    random_seeds = SeedSequence(seed).spawn(n_iterations)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)

    correlation_method = _get_correlation_function(correlation)

    eigenvalue_array = np.zeros((n_iterations, n_items))

    for ndx, rseed in enumerate(random_seeds):
        rng_local = np.random.default_rng(rseed)

        new_data = rng_local.permutation(raw_data, axis=1).reshape(n_items, -1)
        local_correlation = correlation_method(new_data)

        eigenvals = np.linalg.eigvalsh(local_correlation)[::-1]

        eigenvalue_array[ndx] = eigenvals
    
    return eigenvalue_array.mean(0), eigenvalue_array.std(0, ddof=1)


def parallel_analysis(raw_data, n_iterations, correlation=('pearsons',), 
                      seed=None, num_processors=2):
    """Estimate dimensionality from random data permutations.

    Args:
        raw_data:  [n_items x n_observations] Raw collected data
        n_iterations:  Number of iterations to run
        correlation: Method to construct correlation matrix either:
                        ('pearsons',) for continuous data
                        ('polychoric', min_val, max_val) for ordinal data
                        min_val and max_val are the range for the ordinal data
        seed:  (integer) Random number generator seed value
        num_processors: number of processors on a multi-core cpu to use

    Returns
        eigs: mean eigenvalues
        sigma: standard deviation of eigenvalues
    """
    if num_processors == 1:
        return parallel_analysis_serial(raw_data, n_iterations, correlation, seed)
    
    # Get Seeds for repeatablity
    random_seeds = SeedSequence(seed).spawn(n_iterations)
    chunk_seeds = np.array_split(random_seeds, num_processors)

    n_items = raw_data.shape[0]
    raw_data = raw_data.reshape(1, -1)
    
    # Do the parallel calculation
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=raw_data.nbytes)
        shared_buff = np.ndarray(raw_data.shape, 
                                 dtype=raw_data.dtype, buffer=shm.buf)
        shared_buff[:] = raw_data[:]

        with futures.ThreadPoolExecutor(max_workers=num_processors) as pool:
            results = pool.map(_pa_engine, repeat(shm.name), repeat(correlation),
                               repeat(n_items), repeat(raw_data.dtype), 
                               repeat(raw_data.shape), chunk_seeds)

    eigenvalue_array = np.concatenate(list(results), axis=0)

    return eigenvalue_array.mean(0), eigenvalue_array.std(0, ddof=1)


def _pa_engine(name, correlation, n_items, dtype, shape, subset):
    """Parallel analysis engine for distributed computing."""
    correlation_method = _get_correlation_function(correlation)
    eigenvalue_array = np.zeros((subset.shape[0], n_items))
    
    # Read the shared memory buffer
    existing_shm = shared_memory.SharedMemory(name=name)    
    raw_data = np.ndarray(shape, dtype=dtype, 
                          buffer=existing_shm.buf)

    for ndx, rseed in enumerate(subset):
        rng_local = np.random.default_rng(rseed)

        new_data = rng_local.permutation(raw_data, axis=1).reshape(n_items, -1)

        local_correlation = correlation_method(new_data)

        eigenvals = np.linalg.eigvalsh(local_correlation)[::-1]

        eigenvalue_array[ndx] = eigenvals       
        
    return eigenvalue_array    
