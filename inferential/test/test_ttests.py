import unittest

import numpy as np

from RyStats.inferential import (unequal_variance_ttest, equal_variance_ttest, 
                                 one_sample_ttest, repeated_ttest)

from RyStats.inferential.ttests import _p_value_and_confidence_intervals                                 


class TestEqualVariance(unittest.TestCase):
    """Test Fixture for Equal Variance TTest."""

    def test_equal_variance_two_tailed(self):
        """Testing equal variance."""
        rng = np.random.default_rng(49045463547)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(10, 2, 200)
            data2 = rng.normal(10, 2, 200)

            ttest = equal_variance_ttest(data1, data2)

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_equal_variance_left_tailed(self):
        """Testing equal variance."""
        rng = np.random.default_rng(734433186)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 400)
            data2 = rng.normal(2, 1, 400)

            ttest = equal_variance_ttest(data1, data2, 'left')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_equal_variance_right_tailed(self):
        """Testing equal variance."""
        rng = np.random.default_rng(987131781)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 400)
            data2 = rng.normal(2, 1, 400)

            ttest = equal_variance_ttest(data1, data2, 'right')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)


class TestUnEqualVariance(unittest.TestCase):
    """Test Fixture for UnEqual Variance TTest."""

    def test_unequal_variance_two_tailed(self):
        """Testing unequal variance two tailed."""
        rng = np.random.default_rng(135481321)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(10, 2, 200)
            data2 = rng.normal(10, 2, 200)

            ttest = unequal_variance_ttest(data1, data2)

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_unequal_variance_left_tailed(self):
        """Testing unequal variance left tailed."""
        rng = np.random.default_rng(324851351)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 100)
            data2 = rng.normal(2, 1, 100)

            ttest = unequal_variance_ttest(data1, data2, 'left')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_unequal_variance_right_tailed(self):
        """Testing unequal variance right tailed."""
        rng = np.random.default_rng(887943278)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 100)
            data2 = rng.normal(2, 1, 100)

            ttest = unequal_variance_ttest(data1, data2, 'right')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)                  


class TestRepeated(unittest.TestCase):
    """Test Fixture for Repeated TTest."""

    def test_repeated_two_tailed(self):
        """Testing repeated two tailed."""
        rng = np.random.default_rng(6464584234)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(10, 2, 100)
            data2 = data1 + rng.normal(0, .02, 100)

            ttest = repeated_ttest(data1, data2)

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_repeated_left_tailed(self):
        """Testing repeated left tailed."""
        rng = np.random.default_rng(734516519)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 100)
            data2 = data1 + rng.normal(0, .02, 100)

            ttest = repeated_ttest(data1, data2, 'left')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_repeated_right_tailed(self):
        """Testing repeated right tailed."""
        rng = np.random.default_rng(3571954324)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(2, 1, 100)
            data2 = data1 + rng.normal(0, .02, 100)

            ttest = repeated_ttest(data1, data2, 'right')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)

    def test_unequal_sample_size(self):
        """Testing bad inputs."""
        with self.assertRaises(AssertionError):
            repeated_ttest(np.ones((10,)), np.ones((12)))


class TestOneSample(unittest.TestCase):
    """Test Fixture for One-Sample TTest."""

    def test_onesample_two_tailed(self):
        """Testing onesample two tailed."""
        rng = np.random.default_rng(13489132474)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(-5, 2, 100)

            ttest = one_sample_ttest(data1, -5)

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_onesample_left_tailed(self):
        """Testing onesample left tailed."""
        rng = np.random.default_rng(9876138761251)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(15, 1, 100)

            ttest = one_sample_ttest(data1, 15, 'left')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)        

    def test_one_sample_right_tailed(self):
        """Testing onesample right tailed."""
        rng = np.random.default_rng(615419864354)

        passed = 0
        n_iter = 500
        for _ in range(n_iter):
            data1 = rng.normal(12.2, 1, 100)

            ttest = one_sample_ttest(data1, 12.2, 'right')

            if ttest['P_value'] < .05:
                passed +=1

        self.assertAlmostEqual(passed / n_iter, .05, delta=.01)



class TestMiscTest(unittest.TestCase):
    """Test Fixture for random ttests."""

    def test_fail_tailed_option(self):
        """Testing bad tailed option."""

        with self.assertRaises(ValueError):
            _p_value_and_confidence_intervals(2.3, 100, 'greater')

    def test_confidence_intervals(self):
        """Testing the confidence interval test."""
        # Taken from a T-Test table

        # Two Tailed
        p, ci = _p_value_and_confidence_intervals(2.228, 10, 'two')

        self.assertAlmostEqual(p, .05, delta = .001)
        self.assertTrue(ci.shape == (2, ))
        np.testing.assert_allclose(ci, [-2.228, 2.228], atol=.001)

        # Left One-Tailed
        p, ci = _p_value_and_confidence_intervals(1.895, 7, 'left')

        self.assertAlmostEqual(p, .05, delta = .001)
        self.assertTrue(ci.shape == (2, ))
        self.assertTrue(np.isinf(ci[0]))
        np.testing.assert_allclose(ci, [-np.inf, 1.895], atol=.001)

        # Right One-Tailed
        p, ci = _p_value_and_confidence_intervals(1.761, 14, 'right')

        self.assertAlmostEqual(1-p, .05, delta = .001)
        self.assertTrue(ci.shape == (2, ))
        self.assertTrue(np.isinf(ci[1]))        
        np.testing.assert_allclose(ci, [-1.761, np.inf], atol=.001)



if __name__ == """__main__""":
    unittest.main()