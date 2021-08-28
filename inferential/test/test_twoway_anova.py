import unittest

import numpy as np

from RyStats.inferential import twoway_anova


class TestTwoWayAnova(unittest.TestCase):
    """Test fixture for two way anova."""

    def test_two_way_anova_balanced(self):
        """Testing a balanced two-way anova."""
        rng = np.random.default_rng(234098587092870983787785234)
        passed_1 = 0
        passed_2 = 0
        passed_12 = 0
        n_iter = 1000

        level1 = np.zeros((1500,))
        level1[750:] = 1
        level2 = np.array([0, 1, 2] * 500)

        for _ in range(n_iter):
            dataset = rng.standard_normal(1500)
            results = twoway_anova(level1, level2, dataset)

            if results['Type3']['p1'] < 0.05:
                passed_1 +=1 

            if results['Type3']['p2'] < 0.05:
                passed_2 +=1 

            if results['Type3']['p12'] < 0.05:
                passed_12 +=1 
            
        self.assertAlmostEqual(passed_1 / n_iter, 0.05, delta=0.01)
        self.assertAlmostEqual(passed_2 / n_iter, 0.05, delta=0.01)
        self.assertAlmostEqual(passed_12 / n_iter, 0.05, delta=0.01)

        # In a balanced design, all of these should be equal
        for key in results['Type1_a']:
            self.assertAlmostEqual(results['Type1_a'][key],
                                   results['Type1_b'][key])

            self.assertAlmostEqual(results['Type1_a'][key],
                                   results['Type2'][key])

            self.assertAlmostEqual(results['Type1_a'][key],
                                   results['Type3'][key])

        self.assertAlmostEqual(results['grand'],
                               results['weighted_mean'])                                    


    def test_two_way_anova_unbalanced(self):
        """Testing an unbalanced two-way anova."""
        rng = np.random.default_rng(547298718726308975091293)
        passed_1 = 0
        passed_2 = 0
        passed_12 = 0
        n_iter = 1000

        level1 = np.zeros((1500,))
        level1[850:] = 1
        level2 = np.array([0, 1, 2] * 500)

        for _ in range(n_iter):
            dataset = rng.standard_normal(1500)
            results = twoway_anova(level1, level2, dataset)

            if results['Type1_a']['p1'] < 0.05:
                passed_1 +=1 

            if results['Type1_a']['p2'] < 0.05:
                passed_2 +=1 

            if results['Type1_a']['p12'] < 0.05:
                passed_12 +=1 
 
        self.assertAlmostEqual(passed_1 / n_iter, 0.05, delta=0.01)
        self.assertAlmostEqual(passed_2 / n_iter, 0.05, delta=0.01)
        self.assertAlmostEqual(passed_12 / n_iter, 0.05, delta=0.01)

        # In an unbalanced design, these should be different
        for key in ['F1', 'F2']:
            self.assertNotAlmostEqual(results['Type1_a'][key],
                                      results['Type1_b'][key], delta=.0001)

        # Some of these should be the same
        self.assertAlmostEqual(results['Type1_a']['F12'],
                               results['Type1_b']['F12'])

        self.assertAlmostEqual(results['Type1_a']['F12'],
                               results['Type2']['F12'])

        self.assertAlmostEqual(results['Type1_a']['F12'],
                               results['Type3']['F12'])

        self.assertAlmostEqual(results['Type2']['F2'],
                               results['Type1_a']['F2'])

        self.assertAlmostEqual(results['Type2']['F1'],
                               results['Type1_b']['F1'])

        self.assertNotAlmostEqual(results['Type3']['F1'],
                                 results['Type1_b']['F1'])
        
        self.assertNotAlmostEqual(results['Type3']['F1'],
                                  results['Type1_a']['F1'])                               

        self.assertNotAlmostEqual(results['Type3']['F1'],
                                  results['Type1_b']['F1'])                               

        self.assertNotAlmostEqual(results['Type3']['F1'],
                                  results['Type2']['F1'])                               

        self.assertNotAlmostEqual(results['Type3']['F1'],
                                  results['Type2']['F1']) 
                                  
        self.assertNotAlmostEqual(results['grand'],
                                  results['weighted_mean']) 
                                                                

if __name__ == "__main__":
    unittest.main()