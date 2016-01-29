'''
Created on 17/07/2014

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import unittest
from wetb.hawc2.pc_file import PCFile


import numpy as np



class TestPCFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


    def test_PCFile_ae(self):
        pc = PCFile(self.testfilepath + "NREL_5MW_pc.txt", self.testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(pc.thickness(32), 23.78048780487805)
        self.assertEqual(pc.chord(32), 3.673)
        self.assertEqual(pc.pc_set_nr(32), 1)

    def test_PCFile2(self):
        pc = PCFile(self.testfilepath + "NREL_5MW_pc.txt", self.testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(pc.CL(36, 10), 1.358)
        self.assertEqual(pc.CD(36, 10), 0.0255)
        self.assertEqual(pc.CM(36, 10), -0.1103)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
