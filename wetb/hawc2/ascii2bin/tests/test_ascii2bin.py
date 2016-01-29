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

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


class TextUI(object):
    def show_message(self, m):
        pass

    def exec_long_task(self, text, allow_cancel, task, *args, **kwargs):
        return task(*args, **kwargs)

class TestAscii2Bin(unittest.TestCase):


    def testAscii2bin(self):

        for f in ["Hawc2ascii_bin.sel", "Hawc2ascii_bin.dat"]:
            if os.path.exists(testfilepath + f):
                os.remove(testfilepath + f)
        ascii2bin(testfilepath + "Hawc2ascii.sel", ui=TextUI())

        ascii_file = Hawc2io.ReadHawc2(testfilepath + 'Hawc2ascii')
        bin_file = Hawc2io.ReadHawc2(testfilepath + "Hawc2ascii_bin")

        np.testing.assert_array_almost_equal(ascii_file.ReadAscii(), bin_file.ReadBinary(), 1)
        self.assertEqual(ascii_file.ChInfo, bin_file.ChInfo)

    def testAscii2bin_new_name(self):
        for f in ["Hawc2bin.sel", "Hawc2bin.dat"]:
            if os.path.exists(testfilepath + f):
                os.remove(testfilepath + f)
        ascii2bin(testfilepath + "Hawc2ascii.sel", testfilepath + "Hawc2bin.sel", ui=TextUI())

        ascii_file = Hawc2io.ReadHawc2(testfilepath + 'Hawc2ascii')
        bin_file = Hawc2io.ReadHawc2(testfilepath + "Hawc2bin")

        np.testing.assert_array_almost_equal(ascii_file.ReadAscii(), bin_file.ReadBinary(), 1)
        self.assertEqual(ascii_file.ChInfo, bin_file.ChInfo)


    def testSizeOfFile(self):
        self.assertEqual(size_from_file(testfilepath + "Hawc2ascii.sel"), (800, 28))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
