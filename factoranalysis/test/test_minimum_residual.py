import unittest

import numpy as np

from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import minres_factor_analysis as mfa

from RyStats.common import procrustes_rotation


class TestMaximumLikelihood(unittest.TestCase):
    """Test fixture for minimum residual factor analysis."""

    #TODO: Need algorithm validity test

    def test_minimum_residual_recovery(self):
        """Testing Minimum Residual Recovery."""
        rng = np.random.default_rng(49432132341221348721323123324)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, 2, size=10)

        # Create 3 Factor Data
        cor_matrix = np.cov(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) *.5
        loadings_paf, _, variance = mfa(cor_matrix2, 3, initial_guess=initial_guess)

        # Remove any rotation
        rotation = procrustes_rotation(loadings, loadings_paf)
        updated_loadings = loadings_paf @ rotation
        updated_eigs = np.square(updated_loadings).sum(0)

        # Did I Recover initial values (upto a rotation)
        np.testing.assert_allclose(loadings, updated_loadings, atol=1e-3)
        np.testing.assert_allclose(eigenvalues, updated_eigs, atol=1e-3)
        np.testing.assert_allclose(unique_var, variance, atol=1e-3)

    def test_minimum_residual_recovery2(self):
        """Testing Minimum Residual Recovery no initial guess."""
        rng = np.random.default_rng(498556324111616321324125213)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, 2, size=10)

        # Create 3 Factor Data
        cor_matrix = np.cov(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) *.5
        loadings_paf, _, variance = mfa(cor_matrix2, 3)

        # Remove any rotation
        rotation = procrustes_rotation(loadings, loadings_paf)
        updated_loadings = loadings_paf @ rotation
        updated_eigs = np.square(updated_loadings).sum(0)

        # Did I Recover initial values (upto a rotation)
        np.testing.assert_allclose(loadings, updated_loadings, atol=1e-3)
        np.testing.assert_allclose(eigenvalues, updated_eigs, atol=1e-3)
        np.testing.assert_allclose(unique_var, variance, atol=1e-3)        

if __name__ == "__main__":
    unittest.main()