"""Test wetb.hawc2.pc_file
"""
from wetb.hawc2.ae_file import AEFile

import os
import unittest
import tempfile

import numpy as np

from wetb.hawc2.pc_file import PCFile


class TestPCFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path

    def test_PCFile_ae(self):
        """Verify correct values in values of object loaded from pc file"""
        pc = PCFile(self.testfilepath + "NREL_5MW_pc.txt")
        ae = AEFile(self.testfilepath + "NREL_5MW_ae.txt")
        thickness = ae.thickness(36)
        self.assertEqual(pc.CL(thickness, 10), 1.358)
        self.assertEqual(pc.CD(thickness, 10), 0.0255)
        self.assertEqual(pc.CM(thickness, 10), -0.1103)

    def test_write_PCFile(self):
        """Round-trip loading and saving a pc file
        """
        pc1 = PCFile(self.testfilepath + 'NREL_5MW_pc.txt')
        with tempfile.TemporaryDirectory() as tdir:
            pc1.save(tdir + '/test_pc.txt')
            pc2 = PCFile(pc1.filename)
            self.assertEqual(str(pc1), str(pc2))

        tc1, pcs1 = pc1.pc_sets[1]
        tc2, pcs2 = pc2.pc_sets[1]
        np.testing.assert_array_almost_equal(tc1, tc2)
        for pc1, pc2 in zip(pcs1, pcs2):
            np.testing.assert_array_almost_equal(pc1, pc2)



if __name__ == "__main__":
    unittest.main()
