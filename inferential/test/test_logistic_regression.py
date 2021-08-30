import unittest

import numpy as np
import scipy.stats as ssp
from scipy.special import expit

from RyStats.inferential import logistic_regression


class TestLogisticRegression(unittest.TestCase):
    """Test Fixture for logistic regression."""

    def test_logistic_regression_univariate(self):
        """Test logistic regression when only 1 predictor."""
        rng = np.random.default_rng(93745927495038478659)
        n_iter = 250

        slope = 3.5
        offset = 2.4

        coeffs = 0
        for _ in range(n_iter):
            x = rng.normal(0, 1, (1, 1000))            
            y = slope * x + offset
            y = ssp.bernoulli.rvs(expit(y), random_state=rng)
            
            results = logistic_regression(x, y.squeeze())
            regression_coefficients = results['Regression Coefficients']

            coeffs += regression_coefficients
        
        result = coeffs / n_iter
        
        self.assertAlmostEqual(result[0], offset, delta=.02)
        self.assertAlmostEqual(result[1], slope, delta=.02)

    def test_logistic_regression_multivariate(self):
        """Test logistic regression with several predictors."""
        rng = np.random.default_rng(98843219823126543163464)
        n_iter = 250

        slope1 = 3.5
        slope2 = -1.7
        slope3 = 0.76

        offset = 2.4

        coeffs = 0
        for _ in range(n_iter):
            x1 = rng.normal(0, 1, (1, 2000))
            x2 = rng.normal(0, 1, (1, 2000))
            x3 = rng.normal(0, 1, (1, 2000))
            y = slope1 * x1 + slope2 * x2 + slope3 * x3 + offset
            y = ssp.bernoulli.rvs(expit(y), random_state=rng)
            
            results = logistic_regression(np.vstack((x1, x2, x3)), 
                                          y.squeeze())
            regression_coefficients = results['Regression Coefficients']

            coeffs += regression_coefficients
        
        result = coeffs / n_iter

        self.assertAlmostEqual(result[0], offset, delta=.05)
        self.assertAlmostEqual(result[1], slope1, delta=.05)
        self.assertAlmostEqual(result[2], slope2, delta=.05)
        self.assertAlmostEqual(result[3], slope3, delta=.05)
    
    def test_logistic_regression_with_nan(self):
        "Testing logisitic regression with missing data."
        rng = np.random.default_rng(5315556464556484513652389)
        n_iter = 250

        slope1 = -3.5
        slope2 = 1.7
        slope3 = -0.76

        offset = 4.2

        coeffs = 0
        for _ in range(n_iter):
            x1 = rng.normal(0, 1, (1, 2000))
            x2 = rng.normal(0, 1, (1, 2000))
            x3 = rng.normal(0, 1, (1, 2000))

            missing_mask = rng.uniform(0, 1, 2000) < .05
            x1[0, missing_mask] = np.nan

            missing_mask = rng.uniform(0, 1, 2000) < .05
            x2[0, missing_mask] = np.nan

            missing_mask = rng.uniform(0, 1, 2000) < .05
            x3[0, missing_mask] = np.nan

            y = slope1 * x1 + slope2 * x2 + slope3 * x3 + offset
            mask = ~np.isnan(y)
            y[mask] = ssp.bernoulli.rvs(expit(y[mask]), random_state=rng)
            
            results = logistic_regression(np.vstack((x1, x2, x3)), 
                                          y.squeeze())
            regression_coefficients = results['Regression Coefficients']

            coeffs += regression_coefficients
        
        result = coeffs / n_iter

        self.assertAlmostEqual(result[0], offset, delta=.05)
        self.assertAlmostEqual(result[1], slope1, delta=.05)
        self.assertAlmostEqual(result[2], slope2, delta=.05)
        self.assertAlmostEqual(result[3], slope3, delta=.05)
    
    def test_logistic_regression_bad_input(self):
        """Test logistic regression with bad input."""
        slope = 3.5
        offset = 2.4

        x = np.linspace(-10, 10, 100)       
        y = slope * x + offset
        y = ssp.bernoulli.rvs(expit(y))

        with self.assertRaises(AssertionError):
            logistic_regression(x, y+1)    
        
        
if __name__ == "__main__":
    unittest.main()