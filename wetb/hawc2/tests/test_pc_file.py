'''
Created on 17/07/2014

@author: MMPE
'''
import os
import unittest
from wetb.hawc2.pc_file import PCFile


import numpy as np



class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = "test_files/"


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
