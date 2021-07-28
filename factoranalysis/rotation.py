import numpy as np

from scipy.optimize import basinhopping


__all__ = ['sparsify_loadings']


def sparsify_loadings(loadings, initial_guess=None, alpha=0.5,
                      orthogonal=False, seed=None):
    """Sparsifies loadings into approximately simple structure.

    Args:
        loadings:  input factor loadings (n_items x n_factors)
        initial_guess: (optional) seeding for minimization solver
        alpha:  scalar value that regulates between sparseness of solution
                and condition number of the matrix.  Larger alphas
                give more weight to orthogonal solutions and smaller values
                allow more correlation between the factors.
                Optimal values usually in [.25, 2.0].
        orthogonal: ([False] | True) boolean that requires rotation to remain
            perpendicualar.  If set to True overrides alpha
        seed: [Random number generator seed | Generator] for the basinhopping routine

    Returns:
        rotated_loadings: resuls of the sparsifying process
        bases: Bases vectors defining the new coordiante system in row space
    """
    n_factors = loadings.shape[1]
    if initial_guess is None:
        initial_guess = np.random.rand(n_factors * (n_factors - 1)) * 2 * np.pi

    if orthogonal:
        alpha = 50.

    args = (loadings, n_factors, alpha)
    minimizer_kwargs = {'method': 'SLSQP', 'args': args}
    result = basinhopping(_sparse_min_func, initial_guess, niter_success=10,
                          minimizer_kwargs=minimizer_kwargs, seed=seed)

    bases = _oblique_matrix(result['x'], n_factors)
    return loadings @ np.linalg.inv(bases), bases


## Private Helper Functions
def _sparse_min_func(the_angles, factor_loadings, n_bases, alpha=1.0):
    """Minimizes row-wise entropy."""
    the_rotation = _oblique_matrix(the_angles, n_bases)
    return (_entropy(factor_loadings @ np.linalg.inv(the_rotation)) +
            alpha * np.linalg.cond(the_rotation))


def _hyperspherical_vector(thetas):
    """Returns a unit vector in a hypersphere.

    Args:
        theta: array-like iterable of angles in radians

    Returns:
        Hyperspherical unit-vector
    """
    vector = np.ones((np.size(thetas) + 1,))
    vector[1:] = np.cumprod(np.sin(thetas))
    vector[:-1] *= np.cos(thetas)
    return vector


def _oblique_matrix(the_angles, n_factors):
    """Converts angles into a matrix of bases."""
    the_angles = the_angles.reshape(n_factors, the_angles.size // n_factors)
    return (np.array([_hyperspherical_vector(the_angles[ndx])
                      for ndx in range(n_factors)]))


def _entropy(the_array, axis=1):
    """Computes a modified entropy value for each row

    Args:
        the_array:  input array of factors
        axis: Axis to perform calculation on
                0: Vertical
                1: Horizontal
                None: Both axes
    Return:
        entropy value
    """
    power = the_array * the_array
    sum_power = power.sum(axis=axis)
    power /= sum_power[:, None]
    return -(power * np.log(power + 1e-23)).sum()