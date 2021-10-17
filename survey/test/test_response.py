import unittest

import numpy as np

from RyStats.survey import reverse_score, cronbach_alpha, mcdonald_omega


INVALID_RESPONSE = -99


class TestSurvey(unittest.TestCase):
    """Tests fixture for the survey functions."""

    def test_reverse_score(self):
        """Testing reverse score function."""

        dataset = np.array([
            [1, 2, 3, 4, 5, INVALID_RESPONSE]
        ])

        result = reverse_score(dataset, np.array([True]), 5, 
                               invalid_response=INVALID_RESPONSE)

        np.testing.assert_equal(result,[[5, 4, 3, 2, 1, INVALID_RESPONSE]])

        result = reverse_score(dataset, np.array([False]), 5, 
                               invalid_response=INVALID_RESPONSE)

        np.testing.assert_equal(result, dataset)

    def test_reverse_score_fail(self):
        """Testing reverse score function."""
        dataset = np.array([
            [1, 2, 3, 4, 5, INVALID_RESPONSE]
        ])

        with self.assertRaises(AssertionError):
            reverse_score(dataset, np.array([True, False]), 5, 
                          invalid_response=INVALID_RESPONSE)

    def test_cronbach_alpha(self):
        """Testing cronbach alpha."""
        rng = np.random.default_rng(435616365426513465612613263218)

        dataset = rng.integers(1, 5, (10, 1000))
        
        # Regression Tests
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .0901, places=4)

        mask = rng.uniform(0, 1, (10, 1000)) < .05
        dataset[mask] = INVALID_RESPONSE
        
        result = cronbach_alpha(dataset, invalid_response=INVALID_RESPONSE)
        self.assertAlmostEqual(result, .1230, places=4)

    def test_cronbach_alpha2(self):
        """Testing cronbach alpha second test."""
        rng = np.random.default_rng(63352413128574135234)

        dataset = rng.integers(1, 5, (9, 1250))
        
        # Regression Tests
        result = cronbach_alpha(dataset)
        self.assertAlmostEqual(result, .0526, places=4)

        mask = rng.uniform(0, 1, (9, 1250)) < .05
        dataset[mask] = INVALID_RESPONSE
        
        result = cronbach_alpha(dataset, invalid_response=INVALID_RESPONSE)
        self.assertAlmostEqual(result, .1184, places=4)

    def test_mcdonald_omega(self):
        """Testing mcdonalds omega."""
        rng = np.random.default_rng(238561287623161322)

        loadings = rng.uniform(.1, 25, 10)
        uniqueness = rng.uniform(.1, 25, 10)

        expected = loadings.sum()**2 / (loadings.sum()**2 + uniqueness.sum())

        result = mcdonald_omega(loadings, uniqueness)
        self.assertEqual(expected, result)

        # Should handle a single dimension
        result = mcdonald_omega(loadings.reshape(1, -1), uniqueness[:, None])
        self.assertEqual(expected, result)

    def test_mcdonald_omega2(self):
        """Testing mcdonalds omega fails."""
        loadings = np.ones((10, 2))
        uniqueness = np.ones(10)

        with self.assertRaises(AssertionError):
            mcdonald_omega(loadings, uniqueness)

        loadings = np.ones(10)
        uniqueness = np.ones((10, 2))

        with self.assertRaises(AssertionError):
            mcdonald_omega(loadings, uniqueness)

        loadings = np.ones((10, 2))
        uniqueness = np.ones((10, 2))

        with self.assertRaises(AssertionError):
            mcdonald_omega(loadings, uniqueness)

if __name__ == "__main__":
    unittest.main()