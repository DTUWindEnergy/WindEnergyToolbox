'''
Created on 11/07/2016

@author: MMPE
'''
import unittest
from wetb.utils.process_exec import unix_filename
import os


class Test(unittest.TestCase):


    def testUnix_filename(self):
        ufn = "WindEnergyToolbox/wetb/hawc2/Hawc2io.py"
        f = os.path.join(os.path.dirname(__file__), r"../../../..\windenergytoolbox/wetb/HAWC2/hawc2io.py")
        self.assertEqual(unix_filename(f)[-len(ufn):], ufn)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUnix_filename']
    unittest.main()
