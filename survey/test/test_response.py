import unittest

import numpy as np

from RyStats.survey import reverse_score, cronbach_alpha


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


if __name__ == "__main__":
    unittest.main()