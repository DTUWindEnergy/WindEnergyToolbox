'''
Created on 03/12/2015

@author: mmpe
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
import numpy as np
import datetime
from wetb.utils.timing import print_time
from wetb.gtsdf.unix_time import to_unix

class TestUnixTime(unittest.TestCase):

    #@print_time
    def r(self, dt):
        return [to_unix(dt) for dt in dt]

    def test_to_unix(self):
        dt = [datetime.datetime(2000, 1, 1, 12, s % 60) for s in np.arange(1000000)]
        self.r(dt)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
