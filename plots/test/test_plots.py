import unittest

import numpy as np
from bokeh.layouts import row

from RyStats.plots import *


class TestPlots(unittest.TestCase):
    """Test suite for plots."""

    # These only consist of smoke tests to make sure the
    # functions run, there is no checking whatsoever
    def setUp(self):
        "Make fake data for all functions."
        rng = np.random.default_rng(564654365)

        self.corr_mat = np.corrcoef(rng.standard_normal((50, 500)))
        self.loadings = rng.uniform(-1, 1, (10, 5))
        self.eigs = np.arange(20, 0, -1)
        self.p_eigs = rng.normal(1, .1, 20)

    def test_correlation_image(self):
        """Testing correlation image output."""
        # Smoke Test #1 Basic Command
        correlation_image(self.corr_mat)

        # Smoke Test #2 Different Colormap
        correlation_image(self.corr_mat, cmap=1)

        # Smoke Test #3 Different labels
        correlation_image(self.corr_mat, labels=[f'{x-10}' for x in range(50)])

    def test_loading_image(self):
        """Testing loadings image."""
        # Smoke Test #1 Basic Command
        loading_image(self.loadings)

        # Smoke Test #2 Different labels
        loading_image(self.loadings, q_labels=[f'{2*x}' for x in range(10)],
                      f_labels=[f'{4*x}' for x in range(5)])

    def test_scree_plot(self):
        """Testing Scree plot."""
        # Smoke Tests #1 basic call
        scree_plot(self.eigs)

        # Smoke Tests # 2 with parallel eigs
        scree_plot(self.eigs, self.p_eigs)

        # Smoke Tests # 3 with parallel eigs, no difference
        scree_plot(self.eigs, self.p_eigs, False)

        # Smoke Tests # 4 with parallel eigs but not crossing point
        scree_plot(self.eigs, self.p_eigs*0 + .1)

    def test_html_loading_table(self):
        "Testing html loading table"
        # Smoke test #1 basic call
        loading_table(self.loadings)

        # Smoke test #2 new labels
        loading_table(self.loadings, q_labels=[f'{2*x}' for x in range(10)], 
                      f_labels=[f'{4*x}' for x in range(5)])

        # Smoke test #3 html output
        output = loading_table(self.loadings, html_only=True)
        self.assertIsInstance(output, str)
        

if __name__ == "__main__":
    unittest.main()