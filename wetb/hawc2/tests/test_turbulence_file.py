'''
Created on 17/07/2014

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from wetb.hawc2.turbulence_file import TurbulenceFile
standard_library.install_aliases()
import os
import unittest
from wetb.hawc2.pc_file import PCFile


import numpy as np

tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


class TestPCFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

    # def test_TurbulenceFile(self):
    #    TurbulenceFile(tfp + "turb/mann_l29.4_ae1.00_g3.9_h1_64x8x8_1.000x2.00x2.00_s1001u.turb")

    def test_load_from_htc(self):
        u, v, w = TurbulenceFile.load_from_htc(tfp + "htcfiles/test_turb.htc")
        self.assertEqual(u.data.shape, (512, 64))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
