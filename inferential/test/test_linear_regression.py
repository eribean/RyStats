import unittest

import numpy as np

from RyStats.inferential import linear_regression


class TestLinearRegression(unittest.TestCase):
    """Test Fixture for Linear Regression."""

    def test_linear_regression_univariate(self):
        """Test linear regression when only 1 predictor."""
        rng = np.random.default_rng(8763278196128736419)
        x = rng.standard_normal((1, 1000))

        for _ in range(10):
            slope = rng.uniform(-10, 10, 1)
            offset = rng.uniform(-10, 10, 1)
            y = slope * x + offset + rng.normal(0, .1, (1, 1000))

            r_sq = 1 - 1 / (y.var() * y.size * .1)
            
            results = linear_regression(x, y.squeeze())
            regression_coefficients = results['Regression Coefficients']

            self.assertAlmostEqual(results['RSq'], r_sq, delta=.001)
            self.assertAlmostEqual(regression_coefficients[0], offset, delta=.01)
            self.assertAlmostEqual(regression_coefficients[1], slope, delta=.01)

    def test_linear_regression_multivariate(self):
        """Test linear regression when multiple predictors."""
        rng = np.random.default_rng(72872968762908372)
        x1 = rng.standard_normal((1, 1000))
        x2 = rng.standard_normal((1, 1000))
        x3 = rng.standard_normal((1, 1000))

        for _ in range(10):
            slope1 = rng.uniform(-10, 10, 1)
            slope2 = rng.uniform(-10, 10, 1)
            slope3 = rng.uniform(-10, 10, 1)

            offset = rng.uniform(-10, 10, 1)
            y = (slope1 * x1 + slope2 * x2 + slope3 * x3
                 + offset + rng.normal(0, .1, (1, 1000)))
            
            results = linear_regression(np.vstack((x1, x2, x3)), y.squeeze())
            regression_coefficients = results['Regression Coefficients']
            
            self.assertAlmostEqual(regression_coefficients[0], offset, delta=.01)
            self.assertAlmostEqual(regression_coefficients[1], slope1, delta=.01)
            self.assertAlmostEqual(regression_coefficients[2], slope2, delta=.01)
            self.assertAlmostEqual(regression_coefficients[3], slope3, delta=.01)            

    def test_linear_regression_with_nan(self):
        "Testing linear regression with missing data."
        rng = np.random.default_rng(72872968762908372)
        x1 = rng.standard_normal((1, 1000))
        x2 = rng.standard_normal((1, 1000))
        x3 = rng.standard_normal((1, 1000))

        missing_mask = rng.uniform(0, 1, 1000) < .05
        x1[0, missing_mask] = np.nan

        missing_mask = rng.uniform(0, 1, 1000) < .05
        x2[0, missing_mask] = np.nan

        missing_mask = rng.uniform(0, 1, 1000) < .05
        x3[0, missing_mask] = np.nan

        for _ in range(10):
            slope1 = rng.uniform(-10, 10, 1)
            slope2 = rng.uniform(-10, 10, 1)
            slope3 = rng.uniform(-10, 10, 1)

            offset = rng.uniform(-10, 10, 1)
            y = (slope1 * x1 + slope2 * x2 + slope3 * x3
                 + offset + rng.normal(0, .1, (1, 1000)))
            
            results = linear_regression(np.vstack((x1, x2, x3)), y.squeeze())
            regression_coefficients = results['Regression Coefficients']
            
            self.assertAlmostEqual(regression_coefficients[0], offset, delta=.02)
            self.assertAlmostEqual(regression_coefficients[1], slope1, delta=.02)
            self.assertAlmostEqual(regression_coefficients[2], slope2, delta=.02)
            self.assertAlmostEqual(regression_coefficients[3], slope3, delta=.02)   
if __name__ == "__main__":
    unittest.main()