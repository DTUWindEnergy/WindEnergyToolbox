'''
Created on 29/10/2013

@author: mmpe
'''
from wetb.hawc2 import Hawc2io
import numpy as np
import os
import sys
import unittest
from wetb.hawc2.ascii2bin.ascii2bin import ascii2bin, size_from_file

class Test(unittest.TestCase):

    def setUp(self):
        sys.path.append("../")

    def testAscii2bin(self):

        for f in ["Hawc2ascii_bin.sel", "Hawc2ascii_bin.dat"]:
            if os.path.exists(f):
                os.remove(f)
        ascii2bin("Hawc2ascii.sel")

        ascii_file = Hawc2io.ReadHawc2('Hawc2ascii')
        bin_file = Hawc2io.ReadHawc2("Hawc2ascii_bin")

        np.testing.assert_array_almost_equal(ascii_file.ReadAscii(), bin_file.ReadBinary(), 1)
        self.assertEqual(ascii_file.ChInfo, bin_file.ChInfo)

    def testAscii2bin_new_name(self):
        for f in ["Hawc2bin.sel", "Hawc2bin.dat"]:
            if os.path.exists(f):
                os.remove(f)
        ascii2bin("Hawc2ascii.sel", "Hawc2bin.sel")

        ascii_file = Hawc2io.ReadHawc2('Hawc2ascii')
        bin_file = Hawc2io.ReadHawc2("Hawc2bin")

        np.testing.assert_array_almost_equal(ascii_file.ReadAscii(), bin_file.ReadBinary(), 1)
        self.assertEqual(ascii_file.ChInfo, bin_file.ChInfo)


    def testSizeOfFile(self):
        self.assertEqual(size_from_file("hawc2ascii.sel"), (800, 28))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
