import unittest

import numpy as np

from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import minimum_rank_factor_analysis as mrfa


class TestMinimumRank(unittest.TestCase):
    """Test fixture for minimum rank."""

    def test_minimum_rank_recovery(self):
        """Testing Minimum Rank Recovery."""
        rng = np.random.default_rng(2016)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, .5, size=10)

        # Create 3 Factor Data
        cor_matrix = np.corrcoef(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) *.5
        loadings_paf, eigenvalues2, variance = mrfa(cor_matrix2, 3, 
                                                    initial_guess=initial_guess)

        # Did I Recover initial values?
        np.testing.assert_allclose(loadings, loadings_paf, rtol=1e-3)
        np.testing.assert_allclose(eigenvalues, eigenvalues2, rtol=1e-3)
        np.testing.assert_allclose(unique_var, variance, rtol=1e-3)

    def test_minimum_zero_eigenvalue(self):
        """Testing Forced Semi-Positive Definite."""
        rng = np.random.default_rng(12473)

        data = rng.uniform(-2, 2, size=(10, 100))

        # Create 2 Factor Data
        cor_matrix = np.corrcoef(data)

        _, _, variance = mrfa(cor_matrix, 3)

        _, eigens, _ = pca(cor_matrix - np.diag(variance))
        
        # Is the last eigenvalue zero?
        self.assertAlmostEqual(eigens[-1], 0, places=5)


if __name__ == "__main__":
    unittest.main()