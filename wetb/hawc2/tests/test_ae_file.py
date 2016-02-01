'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from wetb.hawc2.ae_file import AEFile
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestAEFile(unittest.TestCase):


    def test_aefile(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(ae.thickness(38.950), 21)
        self.assertEqual(ae.chord(38.950), 3.256)
        self.assertEqual(ae.pc_set_nr(38.950), 1)

    def test_aefile_interpolate(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(ae.thickness(32), 23.78048780487805)
        self.assertEqual(ae.chord(32), 3.673)
        self.assertEqual(ae.pc_set_nr(32), 1)




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
