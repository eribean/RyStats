import unittest

import numpy as np

from RyStats.factoranalysis import principal_components_analysis as pca
from RyStats.factoranalysis import maximum_likelihood_factor_analysis as mlfa
from RyStats.factoranalysis import maximum_likelihood_factor_analysis_em as mlfa_em


from RyStats.common import procrustes_rotation


class TestMaximumLikelihood(unittest.TestCase):
    """Test fixture for maximum likelihood factor analysis."""

    #TODO: Need algorithm validity test

    def test_maximum_likelihood_recovery(self):
        """Testing Maximum Likelihood Recovery Factor."""
        rng = np.random.default_rng(18434)

        data = rng.uniform(-2, 2, size=(10, 100))
        unique_var = rng.uniform(0.2, 2, size=10)

        # Create 3 Factor Data
        cor_matrix = np.cov(data)
        loadings, eigenvalues, _ = pca(cor_matrix, 3)

        # Add Unique variance
        cor_matrix2 = loadings @ loadings.T + np.diag(unique_var)

        initial_guess = np.ones((10,)) *.5
        loadings_paf, _, variance = mlfa(cor_matrix2, 3, initial_guess=initial_guess)

        # Remove any rotation
        rotation = procrustes_rotation(loadings, loadings_paf)
        updated_loadings = loadings_paf @ rotation
        updated_eigs = np.square(updated_loadings).sum(0)

        # Did I Recover initial values (upto a rotation)
        np.testing.assert_allclose(loadings, updated_loadings, rtol=1e-3)
        np.testing.assert_allclose(eigenvalues, updated_eigs, rtol=1e-3)
        np.testing.assert_allclose(unique_var, variance, rtol=1e-3)

    def test_maximum_likelihood_modern_with_classic(self):
        """Testing Maximum Likelihood Modern and Classic equivalence."""        
        rng = np.random.default_rng(78742)

        data = rng.normal(0, 2, size=(20, 5))
        unique_var = rng.uniform(0.2, 2, size=20)

        # Create 3 Factor Data
        cor_matrix = data @ data.T + np.diag(unique_var)#np.cov(data)
         
        loadings_modern, _, variance_modern = mlfa(cor_matrix, 3)
        loadings_classic, _, variance_classic = mlfa_em(cor_matrix, 3)
        loadings_classic2, _, variance_classic2 = mlfa_em(cor_matrix, 3, 
                                                           initial_guess=np.ones((20,)) *.5)

        # Remove any rotation
        rotation = procrustes_rotation(loadings_modern, loadings_classic)
        updated_loadings = loadings_classic @ rotation

        rotation = procrustes_rotation(loadings_modern, loadings_classic2)
        updated_loadings2 = loadings_classic2 @ rotation        

        # No starting Location
        np.testing.assert_allclose(loadings_modern, updated_loadings, rtol=1e-3)
        np.testing.assert_allclose(loadings_modern, updated_loadings2, rtol=1e-3)
        np.testing.assert_allclose(variance_modern, variance_classic, rtol=1e-4)
        np.testing.assert_allclose(variance_modern, variance_classic2, rtol=1e-4)














       



if __name__ == "__main__":
    unittest.main()