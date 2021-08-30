import unittest

import numpy as np

from RyStats.inferential import pearsons_correlation


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

if __name__ == "__main__":
    unittest.main()