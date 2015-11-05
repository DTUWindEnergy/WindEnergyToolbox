'''
Created on 03/09/2015

@author: MMPE
'''
import unittest

from wetb.fast.fast_io import load_output


class Test(unittest.TestCase):


    def test_load_output(self):
        data, info = load_output('testfiles/DTU10MW.out')
        self.assertAlmostEqual(data[4, 3], 4.295E-04)
        self.assertEqual(info['name'], "DTU10MW")
        self.assertEqual(info['attribute_names'][1], "RotPwr")
        self.assertEqual(info['attribute_units'][1], "kW")



    def test_load_binary(self):
        data, info = load_output('testfiles/test_binary.outb')
        self.assertEqual(info['name'], 'test_binary')
        self.assertEqual(info['description'], 'Modified by mwDeriveSensors on 27-Jul-2015 16:32:06')
        self.assertEqual(info['attribute_names'][4], 'RotPwr')
        self.assertEqual(info['attribute_units'][7], 'deg/s^2')
        self.assertAlmostEqual(data[10, 4], 138.822277739535)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testload_output']
    unittest.main()
