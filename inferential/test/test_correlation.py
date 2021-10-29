import unittest

import numpy as np

from RyStats.inferential import pearsons_correlation, polyserial_correlation


class TestCorrelation(unittest.TestCase):
    """Test Fixture for correlation."""

    def test_pearsons_correlation(self):
        """Testing pearsons correlation."""
        rng = np.random.default_rng(34982750394857201981982375)
        n_items = 100
        dataset = rng.standard_normal((n_items, 1000))

        results = pearsons_correlation(dataset)

        # Get the number of valid correlations
        correlation = np.abs(results['Correlation'])
        r_critical = results['R critical']['.05']


        significant_data = (np.count_nonzero(correlation > r_critical) 
                            - n_items) / (n_items * (n_items - 1))

        self.assertAlmostEqual(significant_data, .05, delta=0.01)


class TestPolyserialCorrelation(unittest.TestCase):
    """Polyserial Correlation Test Fixture."""

    def test_polyserial_correlation(self):
        """Testing polyserial corelation function."""
        rng = np.random.default_rng(425365645347626485721532938464553254)
        
        rho = -0.6
        thresholds = [-.2, 0., .8]
        continuous = rng.multivariate_normal([0, 0], [[1, rho], 
                                                       [rho, 1]], size=10000)
        ordinal = np.digitize(continuous[:, 1], thresholds)

        result = polyserial_correlation(continuous[:, 0], ordinal)
        point_polyserial = np.corrcoef(continuous[:, 0], ordinal)[0, 1]

        self.assertAlmostEqual(result, rho, delta=.01)
        self.assertLess(np.abs(result - rho),
                        np.abs(point_polyserial - rho))

    def test_biserial_correlation(self):
        """Testing biserial correlation."""
        # The polyserial function should include binary
        # inputs
        rng = np.random.default_rng(7921354169283445716651382455716656333145)
        
        rho = 0.45
        thresholds = [.3]
        continuous = rng.multivariate_normal([0, 0], [[1, rho], 
                                                       [rho, 1]], size=10000)
        ordinal = np.digitize(continuous[:, 1], thresholds)

        result = polyserial_correlation(continuous[:, 0], ordinal)
        point_polyserial = np.corrcoef(continuous[:, 0], ordinal)[0, 1]

        self.assertAlmostEqual(result, rho, delta=.015)
        self.assertLess(np.abs(result - rho),
                        np.abs(point_polyserial - rho))               

if __name__ == "__main__":
    unittest.main()