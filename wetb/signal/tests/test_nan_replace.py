'''
Created on 02/11/2015

@author: MMPE
'''
import unittest
import numpy as np
from wetb.signal.nan_replace import replace_by_mean, replace_by_line, \
    replace_by_polynomial, max_no_nan
from matplotlib.pyplot import plot, show
class TestNan_replace(unittest.TestCase):


    def test_nan_replace_by_mean(self):
        a = np.array([1, 5, 6, 4, 5, np.nan, 3, 1, 5, 6, 4.])
        np.testing.assert_array_equal(replace_by_mean(a), [1, 5, 6, 4, 5, 4, 3, 1, 5, 6, 4.])
        a = np.array([1, 5, 6, 4, 5, np.nan, np.nan, 1, 5, 6, 4.])
        np.testing.assert_array_equal(replace_by_mean(a), [1, 5, 6, 4, 5, 3, 3, 1, 5, 6, 4.])
        a = np.array([np.nan, 5, 6, 4, 5, np.nan, 3, 1, 5, 6, np.nan])
        np.testing.assert_array_equal(replace_by_mean(a), [5, 5, 6, 4, 5, 4, 3, 1, 5, 6, 6])


    def test_nan_replace_by_line(self):
        a = np.array([1, 5, 6, 4, 5, np.nan, 3, 1, 5, 6, 4.])
        np.testing.assert_array_equal(replace_by_line(a), [1, 5, 6, 4, 5, 4, 3, 1, 5, 6, 4.])
        a = np.array([1, 5, 6, 4, 5, np.nan, np.nan, 2, 5, 6, 4.])
        np.testing.assert_array_equal(replace_by_line(a), [1, 5, 6, 4, 5, 4, 3, 2, 5, 6, 4.])
        a = np.array([np.nan, 5, 6, 4, 5, np.nan, 3, 1, 5, 6, np.nan])
        np.testing.assert_array_equal(replace_by_line(a), [5, 5, 6, 4, 5, 4, 3, 1, 5, 6, 6])


    def test_nan_replace_by_polynomial(self):
        a = np.array([np.nan, 5, 6, 4, 5, np.nan, 3, 1, 5, 6, np.nan])
        np.testing.assert_array_equal(replace_by_polynomial(a), [5, 5, 6, 4, 5, 4, 3, 1, 5, 6, 6])
        a = np.array([1, 5, 6, 4, 5, np.nan, np.nan, 1, 5, 6, 4.])
        np.testing.assert_array_almost_equal(replace_by_polynomial(a, 3, 2), [1, 5, 6, 4, 5, 3.3, 1.2, 1, 5, 6, 4.])

        if 0:
            plot(a)
            plot(replace_by_polynomial(a, 3, 2))
            show()


    def test_nan_replace_by_polynomial2(self):
        a = np.r_[np.arange(10), np.repeat(np.nan, 100), 10 - np.arange(10)]
        self.assertLessEqual(replace_by_polynomial(a, 3, 9).max(), np.nanmax(a))

        if 0:
            plot(a, '.r')
            plot(replace_by_polynomial(a, 3, 9), 'g-')
            show()


    def test_max_no_nan(self):
        a = np.array([1, 5, 6, 4, 5, np.nan, 3, 1, 5, np.nan, 4.])
        self.assertEqual(max_no_nan(a), 1)
        a = np.array([1, 5, 6, 4, 5, np.nan, np.nan, 1, 5, np.nan, 4.])
        self.assertEqual(max_no_nan(a), 2)
        a = np.array([np.nan, np.nan, np.nan, 4, 5, np.nan, np.nan, 1, 5, np.nan, 4.])
        self.assertEqual(max_no_nan(a), 3)
        a = np.array([1, 5, 6, 4, 5, np.nan, 4, 1, 5, np.nan, np.nan])
        self.assertEqual(max_no_nan(a), 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_nanreplace']
    unittest.main()
