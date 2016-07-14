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
from wetb.hawc2.st_file import StFile
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestStFile(unittest.TestCase):


    def test_stfile(self):
        st = StFile(testfilepath + "DTU_10MW_RWT_Blade_st.dat")
        self.assertEqual(st.radius()[2], 3.74238)
        self.assertEqual(st.radius(3), 3.74238)
        self.assertEqual(st.x_e(67.7351), 4.4320990737400E-01)
        self.assertEqual(st.E(3.74238, 1, 1), 1.2511695058500E+10)
        self.assertEqual(st.E(3.74238, 1, 2), 1.2511695058500E+27)


    def test_stfile_interpolate(self):
        st = StFile(testfilepath + "DTU_10MW_RWT_Blade_st.dat")
        self.assertAlmostEqual(st.x_e(72.2261), 0.381148048)
        self.assertAlmostEqual(st.y_e(72.2261), 0.016692967)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
