# -*- coding: utf-8 -*-
"""
Test body_matrix_output_file module.

@author: ricriv
"""

# %% Import.

import os
import unittest
import numpy.testing as npt
from wetb.hawc2.body_matrix_output_file import read_body_matrix_output


# %% Test.

# Test file path.
tfp = os.path.join(os.path.dirname(__file__), "test_files/")


class TestBodyOutput(unittest.TestCase):
    def test_all(self):
        # Read files.
        bodies = read_body_matrix_output(f"{tfp}/body_matrix_output/body_")

        # Have we read all files?
        self.assertEqual(set(bodies.keys()), {"blade1_1", "tower"})
        self.assertEqual(
            set(bodies["tower"].keys()), {"mass", "damping", "stiffness"}
        )
        self.assertEqual(
            set(bodies["blade1_1"].keys()), {"mass", "damping", "stiffness"}
        )

        # Check matrices shape.
        npt.assert_array_equal(bodies["tower"]["mass"].shape, (66, 66))
        npt.assert_array_equal(bodies["blade1_1"]["mass"].shape, (18, 18))

        # Check 1 value.
        npt.assert_almost_equal(
            bodies["tower"]["mass"][0, 0], 627396.222002882
        )


if __name__ == "__main__":
    unittest.main()
