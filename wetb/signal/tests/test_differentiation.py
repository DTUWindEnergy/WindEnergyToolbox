'''
Created on 29. mar. 2017

@author: mmpe
'''
import unittest
import numpy as np
from wetb.signal.filters._differentiation import differentiation

class Test(unittest.TestCase):


    def testDifferentiation(self):
        np.testing.assert_array_equal(differentiation([1,2,1,0,1,1]), [1,0,-1,0,.5,0])
        np.testing.assert_array_equal(differentiation([1,2,1,0,1,1], 'left'), [np.nan, 1,-1,-1,1,0])
        np.testing.assert_array_equal(differentiation([1,2,1,0,1,1], 'right'), [1,-1,-1,1,0, np.nan])
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDifferentiation']
    unittest.main()