'''
Created on 03/12/2015

@author: mmpe
'''
import unittest
import numpy as np
import datetime
from wetb.gtsdf.unix_time import to_unix, from_unix


class TestUnixTime(unittest.TestCase):


    def test_to_unix(self):
        self.assertEqual(to_unix(datetime.datetime(2016, 2, 2, 13, 6, 25)), 1454418385)
        self.assertEqual(to_unix([datetime.datetime(2016, 2, 2, 13, 6, 25),datetime.datetime(2016, 2, 2, 13, 6, 26)]), [1454418385,1454418386])
        self.assertNotEqual(to_unix(datetime.datetime(2016, 2, 2, 13, 6, 26)), 1454418385)
        self.assertRaises(Exception, to_unix,1)


    def test_from_unix(self):
        self.assertEqual(from_unix(1454418385), datetime.datetime(2016, 2, 2, 13, 6, 25))
        self.assertNotEqual(from_unix(1454418385), datetime.datetime(2016, 2, 2, 13, 6, 26))
        
        self.assertEqual(from_unix(np.nan), datetime.datetime(1970,1,1,0,0))
        self.assertEqual(from_unix([1454418385,1454418386]), [datetime.datetime(2016, 2, 2, 13, 6, 25),datetime.datetime(2016, 2, 2, 13, 6, 26)])
        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
