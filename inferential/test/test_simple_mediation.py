import unittest

import numpy as np

from RyStats.inferential import simple_mediation


class TestSimpleMediation(unittest.TestCase):
    """Test Fixture for Simple Mediation."""
    
    def test_total_mediation(self):
        """Testing total mediation."""
        rng = np.random.default_rng(842574782795233252432)

        coeff1 = -1.2
        coeff2 = 2.3

        independent = rng.standard_normal(1000)
        mediator = coeff1 * independent + rng.normal(0, .3, 1000)
        dependent = coeff2 * mediator + rng.normal(0, .2, 1000)

        results = simple_mediation(dependent, independent, mediator)

        self.assertAlmostEqual(results['Mediated Effect']['Coefficient'], coeff2, delta=0.02)
        self.assertAlmostEqual(results['Second Effect']['Coefficient'], coeff1, delta=0.02)
        self.assertAlmostEqual(results['Direct Effect']['Coefficient'], 0.0, delta=0.02)
        self.assertAlmostEqual(results['Percent Mediated']['Coefficient'], 100, delta=1.0)
        
    def test_no_mediation(self):
        """Testing no mediation."""
        rng = np.random.default_rng(62098271062615234511)

        coeff1 = -1.2
        coeff2 = 2.3

        independent = rng.standard_normal(1000)
        mediator = coeff1 * independent + rng.normal(0, .3, 1000)
        dependent = coeff2 * independent + rng.normal(0, .2, 1000)

        results = simple_mediation(dependent, independent, mediator)

        self.assertAlmostEqual(results['Mediated Effect']['Coefficient'], 0.0, delta=0.02)
        self.assertAlmostEqual(results['Second Effect']['Coefficient'], coeff1, delta=0.02)
        self.assertAlmostEqual(results['Direct Effect']['Coefficient'], coeff2, delta=0.02)
        self.assertAlmostEqual(results['Percent Mediated']['Coefficient'], 0, delta=1.0)

    def test_partial_mediation(self):
        """Testing partial mediation."""
        rng = np.random.default_rng(62098271062615234511)

        coeff1 = 1.2
        coeff2 = 2.3
        coeff3 = 0.76

        independent = rng.standard_normal(1000)
        mediator = coeff1 * independent + rng.normal(0, .3, 1000)
        dependent = coeff2 * mediator + coeff3 * independent + rng.normal(0, .2, 1000)

        results = simple_mediation(dependent, independent, mediator)

        self.assertAlmostEqual(results['Mediated Effect']['Coefficient'], 
                               coeff2, delta=0.02)
        self.assertAlmostEqual(results['Second Effect']['Coefficient'], coeff1, delta=0.02)
        self.assertAlmostEqual(results['Direct Effect']['Coefficient'], coeff3, delta=0.02)

        percent_mediated = 100 * (coeff1 * coeff2 / (coeff3 + coeff1 * coeff2))
        self.assertAlmostEqual(results['Percent Mediated']['Coefficient'], 
                               percent_mediated, delta=1.0)


if __name__ == "__main__":
    unittest.main()