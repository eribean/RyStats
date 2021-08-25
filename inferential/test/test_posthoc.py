import unittest

import numpy as np

from RyStats.inferential import tukey_posthoc, games_howell_posthoc, bonferonni_posthoc


class TestPostHocMethods(unittest.TestCase):
    """Test Fixture for Post Hoc Methods."""

    def setUp(self):
        rng = np.random.default_rng(3439857390857420398471098273)
        self.dataset = [rng.standard_normal(250) for _ in range(4)]
        self.dataset2 = [rng.normal(0, 1 + ndx/4, 250) for ndx in range(4)]

    def test_tukey_posthoc(self):
        """Testing Tukey Post-Hoc."""     
        rr, cc = np.tril_indices(4, k=-1)

        result = tukey_posthoc(*self.dataset)
        
        # Regression Test
        expected_p_value = np.array([0.93949605, 0.67821687, 0.9483563, 
                                     0.99940972, 0.96826012, 0.74926386])

        expected_lower_ci = np.array([-0.29021765, -0.34029182, -0.28727278, 
                                      -0.24809237, -0.19507333, -0.14499915])

        np.testing.assert_allclose(result['P_value'][rr, cc], 
                                   expected_p_value, atol=1e-4)
        np.testing.assert_allclose(result['Lower CI'][rr, cc], 
                                   expected_lower_ci, atol=1e-4)

    def test_games_howell_posthoc(self):
        """Testing Games Howell Post Hoc."""
        rr, cc = np.tril_indices(4, k=-1)

        result = games_howell_posthoc(*self.dataset)
        
        # Regression Test
        expected_p_value = np.array([0.942161, 0.689703, 0.945025, 
                                     0.999446, 0.96665 , 0.739303])

        expected_lower_ci = np.array([-0.294567, -0.344686, -0.282424, 
                                      -0.253633, -0.191418, -0.14139])

        np.testing.assert_allclose(result['P_value'][rr, cc], 
                                   expected_p_value, atol=1e-4)
        np.testing.assert_allclose(result['Lower CI'][rr, cc], 
                                   expected_lower_ci, atol=1e-4)

    def test_equal_bonferonni_posthoc(self):
        """Testing Bonferonni Post Hoc Equal."""
        rr, cc = np.tril_indices(4, k=-1)

        result = bonferonni_posthoc(*self.dataset2, ttest_type='equal')
        
        # Regression Test
        expected_p_value = np.array([0.237767, 0.561773, 0.717279, 
                                     0.84575 , 0.295532, 0.533447])

        expected_lower_ci = np.array([-0.303003, -0.298203, -0.20255, 
                                      -0.217053, -0.120646, -0.197812])

        np.testing.assert_allclose(result['P_value'][rr, cc], 
                                   expected_p_value, atol=1e-5)
        np.testing.assert_allclose(result['Lower CI'][rr, cc], 
                                   expected_lower_ci, atol=1e-5)

    def test_unequal_bonferonni_posthoc(self):
        """Testing Games Howell Post Hoc UnEqual."""
        rr, cc = np.tril_indices(4, k=-1)

        result = bonferonni_posthoc(*self.dataset2, ttest_type='unequal')
        
        # Regression Test
        expected_p_value = np.array([0.2378  , 0.561835, 0.717291, 
                                     0.845772, 0.295589, 0.533448])

        expected_lower_ci = np.array([-0.303031, -0.298337, -0.202597, 
                                      -0.21722 , -0.120716, -0.197815])

        np.testing.assert_allclose(result['P_value'][rr, cc], 
                                   expected_p_value, atol=1e-4)
        np.testing.assert_allclose(result['Lower CI'][rr, cc], 
                                   expected_lower_ci, atol=1e-4)

    def test_repeated_bonferonni_posthoc(self):
        """Testing Games Howell Post Hoc Repeated."""
        rr, cc = np.tril_indices(4, k=-1)

        result = bonferonni_posthoc(*self.dataset2, ttest_type='repeated')
        
        # Regression Test
        expected_p_value = np.array([0.228903, 0.551281, 0.716359, 
                                     0.843387, 0.310797, 0.545441])

        expected_lower_ci = np.array([-0.299669, -0.292557, -0.202171, 
                                      -0.213828, -0.12932 , -0.207034])

        np.testing.assert_allclose(result['P_value'][rr, cc], 
                                   expected_p_value, atol=1e-4)
        np.testing.assert_allclose(result['Lower CI'][rr, cc], 
                                   expected_lower_ci, atol=1e-4)                                                                        

if __name__ == "__main__":
    unittest.main()