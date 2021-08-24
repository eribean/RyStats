import unittest
import numpy as np

from RyStats.inferential import (equal_variance_oneway_anova,
                                 unequal_variance_oneway_anova,
                                 repeated_oneway_anova)


class TestOneWayAnova(unittest.TestCase):
    """Test Fixture for 1-way Anova Tests."""

    def test_equal_variance(self):
        """Testing 1-way equal variance."""
        rng = np.random.default_rng(3497225439486967230)

        passed = 0
        n_iter = 1000

        for _ in range(n_iter):
            # Create data
            dataset = [rng.standard_normal(1000) for _ in range(4)]

            result = equal_variance_oneway_anova(*dataset)

            if result["P_value"] < 0.05:
                passed += 1
        
        self.assertAlmostEqual(passed / n_iter, 0.05, delta=.01)

    def test_unequal_variance(self):
        """Testins 1-way unequal variance."""
        rng = np.random.default_rng(9879520934516587512093487)

        passed = 0
        n_iter = 1000

        for _ in range(n_iter):
            # Create data
            dataset = [rng.normal(0, 1 + ndx/2, 1000) 
                       for ndx in range(4)]

            result = unequal_variance_oneway_anova(*dataset)

            if result["P_value"] < 0.05:
                passed += 1
        
        self.assertAlmostEqual(passed / n_iter, 0.05, delta=.01)

    def test_repeated_measure(self):
        """Testing 1-way repeated measures."""
        rng = np.random.default_rng(9879520934516587512093487)

        passed = 0
        n_iter = 1000
        base_data = rng.standard_normal(100)

        for _ in range(n_iter):
            # Create data
            dataset = [base_data + rng.normal(0, 0.1, 100) 
                       for ndx in range(4)]

            result = repeated_oneway_anova(*dataset)

            if result["P_value"] < 0.05:
                passed += 1
        
        self.assertAlmostEqual(passed / n_iter, 0.05, delta=.01)    
        
    def test_repeated_measure_fail(self):
        """Testing 1-way repeated measures."""
        dataset = [np.ones(100), np.ones(101)]

        with self.assertRaises(AssertionError):
            repeated_oneway_anova(*dataset)

if __name__ == "__main__":
    unittest.main()