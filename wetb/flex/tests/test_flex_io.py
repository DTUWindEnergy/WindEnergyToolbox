'''
Created on 11. apr. 2017

@author: mmpe
'''
import os
import unittest
from wetb import flex


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class Test(unittest.TestCase):


    def test_load(self):
        time, data, info = flex.load(tfp+"test1/test.int")
        self.assertEqual(data.shape, (800,7))
        self.assertEqual(info['attribute_names'][0], "WSP_gl._")
        self.assertEqual(info['attribute_units'][0], "m/s")
        self.assertEqual(info['attribute_descriptions'][0], "Free wind speed Vy, gl. coo, of gl. pos 0.75, 0.00, -40.75")

        self.assertAlmostEqual(data[0, 1], 12.037,3)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()