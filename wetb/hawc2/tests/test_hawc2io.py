'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
import numpy as np
from wetb.hawc2.Hawc2io import ReadHawc2
import os


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/hawc2io/')  # test file path

class TestHAWC2IO(unittest.TestCase):


    def test_doc_example(self):
        # if called with ReadOnly = 1 as
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        # no channels a stored in memory, otherwise read channels are stored for reuse

        # channels are called by a list
#        file([0,2,1,1])  => channels 1,3,2,2
        self.assertEqual(file([0, 2, 1, 1]).shape, (800, 4))

#        # if empty all channels are returned
#        file()  => all channels as 1,2,3,...
        self.assertEqual(file().shape, (800, 28))
#        file.t => time vector
        np.testing.assert_array_almost_equal(file.t, file([0])[:, 0])


    def test_read_binary_file(self):
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)

    def test_read_ascii_file(self):
        file = ReadHawc2(testfilepath + "Hawc2ascii", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
