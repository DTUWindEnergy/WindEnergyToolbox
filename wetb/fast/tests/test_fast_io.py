'''
Created on 03/09/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

from wetb.fast.fast_io import load_output
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestFastIO(unittest.TestCase):


    def test_load_output(self):
        data, info = load_output(testfilepath + 'DTU10MW.out')
        self.assertAlmostEqual(data[4, 3], 4.295E-04)
        self.assertEqual(info['name'], "DTU10MW")
        self.assertEqual(info['attribute_names'][1], "RotPwr")
        self.assertEqual(info['attribute_units'][1], "kW")



    def test_load_binary(self):
        data, info = load_output(testfilepath + 'test_binary.outb')
        self.assertEqual(info['name'], 'test_binary')
        self.assertEqual(info['description'], 'Modified by mwDeriveSensors on 27-Jul-2015 16:32:06')
        self.assertEqual(info['attribute_names'][4], 'RotPwr')
        self.assertEqual(info['attribute_units'][7], 'deg/s^2')
        self.assertAlmostEqual(data[10, 4], 138.822277739535)

    def test_load_output2(self):
        data, info = load_output(testfilepath + 'DTU10MW.out')
        self.assertEqual(info['name'], "DTU10MW")
        self.assertEqual(info['attribute_names'][1], "RotPwr")
        self.assertEqual(info['attribute_units'][1], "kW")



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testload_output']
    unittest.main()
