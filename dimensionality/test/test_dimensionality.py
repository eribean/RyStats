import unittest

import numpy as np

from RyStats.dimensionality import minimum_average_partial, parallel_analysis, parallel_analysis_serial
from RyStats.dimensionality.parallel_analysis import _pa_engine, _get_correlation_function


class TestDimensionality(unittest.TestCase):
    """Test fixture for dimensionality."""

    def test_minimum_average_partial(self):
        """Testing MAP dimensionality."""
        rng = np.random.default_rng(30212303)

        dataset = rng.normal(0, 2, (50 ,300))
        correlation = np.corrcoef(dataset)
        
        eigs, vects = np.linalg.eigh(correlation)
        eigs[:45] /= 50.0
        correlation = vects @ np.diag(eigs) @ vects.T

        result = minimum_average_partial(correlation)
        self.assertEqual(result[0], 5)

        # Regression
        first_ten = np.array([0.00550975, 0.16907593, 0.20150386, 0.24134979, 
                              0.2528665, 0.00531349, 0.0057849, 0.00629515, 
                              0.0068299 , 0.00740974,])
        
        np.testing.assert_allclose(first_ten, result[1][:10], rtol=1e-4)

    def test_parallel_analysis_serial_pearsons(self):
        """Testing parallel analysis serial creation."""
        rng = np.random.default_rng(49435737)

        dataset = rng.normal(0, 2, (50 ,300))
        correlation = np.corrcoef(dataset)
        
        eigs, vects = np.linalg.eigh(correlation)
        eigs[:45] /= 50.0
        dataset = vects @ np.diag(eigs) @ rng.normal(0, 1, size=(50, 300))

        result = parallel_analysis_serial(dataset, 20, correlation=('pearsons',),
                                          seed=None)
        
        better = (eigs - result[0])
        n_found = np.count_nonzero(better > 0)

        self.assertEqual(n_found, 5)

    def test_parallel_analysis_serial_polychoric(self):
        """Testing parallel analysis serial creation with polychoric correlation."""
        rng = np.random.default_rng(41656732198)

        dataset = rng.normal(0, 2, (10 , 300))
        correlation = np.corrcoef(dataset)
        
        eigs, vects = np.linalg.eigh(correlation)
        eigs[:5] /= 20.0
        dataset = vects @ np.diag(eigs) @ rng.normal(0, 1, size=(10, 100))
        dataset_int = np.digitize(dataset, [-.2, .2, 1.3])

        result = parallel_analysis_serial(dataset_int, 5, 
                                          correlation=('polychoric', 0, 4),
                                          seed=None)
        
        better = (eigs - result[0])
        n_found = np.count_nonzero(better > 0)

        self.assertEqual(n_found, 5)

    def test_parallel_analysis_serial_bad_correlation(self):
        """Testing parallel analysis with bad input."""
        with self.assertRaises(ValueError):
            parallel_analysis_serial(np.zeros((10, 40)), 10, 
                                     correlation=('Seahawks',))

    def test_parallel_analysis_parallel_pass_through(self):
        """Testing parallel analysis with 1 processor."""
        rng = np.random.default_rng(5623456234)

        dataset = rng.normal(0, 2, (50, 300))
        correlation = np.corrcoef(dataset)
        
        eigs, vects = np.linalg.eigh(correlation)
        eigs[:45] /= 50.0
        dataset = vects @ np.diag(eigs) @ rng.normal(0, 1, size=(50, 100))

        result = parallel_analysis(dataset, 20, seed=84357242, 
                                   num_processors=1)

        better = (eigs - result[0])
        n_found = np.count_nonzero(better > 0)

        self.assertEqual(n_found, 5)

    def test_parallel_analysis_parallel(self):
        """Testing parallel analysis with 2 processor."""
        rng = np.random.default_rng(42362436)

        dataset = rng.normal(0, 2, (50, 300))
        correlation = np.corrcoef(dataset)
        
        eigs, vects = np.linalg.eigh(correlation)
        eigs[:45] /= 50.0
        dataset = vects @ np.diag(eigs) @ rng.normal(0, 1, size=(50, 100))

        result_parallel = parallel_analysis(dataset, 20, seed=62346243, 
                                            num_processors=2)

        result_serial = parallel_analysis_serial(dataset, 20, seed=62346243)

        # Two results better be equal
        np.testing.assert_equal(result_parallel[0], result_serial[0])
        np.testing.assert_equal(result_parallel[1], result_serial[1])


if __name__ == "__main__":
    unittest.main()