'''
Created on 19/12/2014

@author: MMPE
'''
import unittest

from matplotlib.pyplot  import *
import numpy as np
from wetb import signal
from wetb import gtsdf
import os
from wetb.gtsdf.unix_time import from_unix
import datetime


class TestInterpolation(unittest.TestCase):

    def test_interpolate1(self):
        x = [1, 1.5, 2, 3]
        xp = [0.5, 1.5, 3]
        yp = [5, 15, 30]
        np.testing.assert_array_equal(signal.interpolate(x, xp, yp), [10., 15., 20, 30.])
        np.testing.assert_array_equal(signal.interpolate(x, xp, yp, 1), [10., 15., np.nan, 30.])




    def test_interpolate2(self):
        x = np.arange(0, 100, 5, dtype=float)
        xp = np.arange(0, 100, 10, dtype=float)
        xp = np.r_[xp[:3], xp[5:]]
        yp = np.arange(10, dtype=float)
        yp[7:8] = np.nan
        yp = np.r_[yp[:3], yp[5:]]

        #x = [ 0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.  60.  65.  70.  75.  80.  85.  90.  95.]
        #xp = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        #yp = [0.0, 1.0, 2.0, 5.0, 6.0, nan, 8.0, 9.0]

        y = signal.interpolate(x, xp, yp)
        np.testing.assert_array_equal(y[~np.isnan(y)], [0., 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 8. , 8.5, 9. ])

        y = signal.interpolate(x, xp, yp, 10)
        np.testing.assert_array_equal(y[~np.isnan(y)], [ 0. , 0.5, 1. , 1.5, 2. , 5. , 5.5, 8. , 8.5, 9. ])



    def test_interpolate_max_dydxp(self):
        x = np.arange(7)
        xp = [0, 2, 4, 6]
        yp = [358, 359, 0, 1]
        y = signal.interpolate(x, xp, yp, max_dydxp=30)
        np.testing.assert_array_equal(y, [ 358., 358.5, 359., np.nan, 0., 0.5, 1. ])

        y = signal.interpolate(x, xp, yp, max_dydxp=180)
        np.testing.assert_array_equal(y, [ 358., 358.5, 359., 179.5, 0., 0.5, 1. ])


    def test_interpolate_max_dydxp_cyclic(self):
        x = np.arange(7)
        xp = [0, 2, 4, 6]
        yp = [358, 359, 0, 1]

        y = signal.interpolate(x, xp, [178, 179, -180, -179], max_dydxp=30, cyclic_range=360)
        np.testing.assert_array_equal(y, [ 178. , 178.5, 179. , -0.5, -180. , -179.5, -179. ])

        y = signal.interpolate(x, xp, yp, max_dydxp=30, cyclic_range=360)
        np.testing.assert_array_equal(y, [ 358., 358.5, 359., 359.5, 0., 0.5, 1. ])

        y = signal.interpolate(xp, x, [ 358., 358.5, 359., 359.5, 0., 0.5, 1. ], max_dydxp=30, cyclic_range=360)
        np.testing.assert_array_equal(y, [ 358., 359., 0., 1. ])

        y = signal.interpolate(x, xp, yp, max_dydxp=180)
        np.testing.assert_array_equal(y, [ 358., 358.5, 359., 179.5, 0., 0.5, 1. ])

    def test_interpolate_max_repeated(self):
        x = np.arange(7)
        xp = [0, 1, 2, 3, 4, 5, 6]
        yp = [4, 5, 5, 5, 6, 6, 7]
        y = signal.interpolate(x, xp, yp, max_repeated=2)
        np.testing.assert_array_equal(y, [ 4, 5, np.nan, np.nan, 6, 6, 7])


        x = np.arange(7)
        xp = [0, 3, 4, 5, 6]
        yp = [5, 5, 7, 6, 6]
        y = signal.interpolate(x, xp, yp, max_repeated=2)
        np.testing.assert_array_equal(y, [ 5, np.nan, np.nan, np.nan, 7, 6, 6])

        xp = [0, 2, 3, 4, 6]
        y = signal.interpolate(x, xp, yp, max_repeated=1)
        np.testing.assert_array_equal(y, [ 5, np.nan, np.nan, 7, 6, np.nan, np.nan])


        xp = [0, 1, 2, 3, 6]
        y = signal.interpolate(x, xp, yp, max_repeated=2)
        np.testing.assert_array_equal(y, [ 5, 5, 7, 6, np.nan, np.nan, np.nan])



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
