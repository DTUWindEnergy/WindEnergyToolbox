'''
Created on 11/07/2016

@author: MMPE
'''
import unittest
from wetb.utils.process_exec import unix_filename
import os


class Test(unittest.TestCase):


    def testUnix_filename(self):
        print (unix_filename(os.path.dirname(__file__) + r"../../../..\windenergytoolbox/wetb/HAWC2/hawc2io.py"))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUnix_filename']
    unittest.main()