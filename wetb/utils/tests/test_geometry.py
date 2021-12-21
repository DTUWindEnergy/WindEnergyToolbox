'''
Created on 15/01/2014

@author: MMPE
'''
import unittest


import numpy as np
from wetb.utils.geometry import rad, deg, mean_deg, sind, cosd, std_deg, tand,\
    rpm2rads, rads2rpm
import os


class TestGeometry(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path

    def test_rad(self):
        self.assertEqual(rad(45), np.pi / 4)
        self.assertEqual(rad(135), np.pi * 3 / 4)


    def test_deg(self):
        self.assertEqual(45, deg(np.pi / 4))
        self.assertEqual(135, deg(np.pi * 3 / 4))

    def test_rad_deg(self):
        for i in [15, 0.5, 355, 400]:
            self.assertEqual(i, deg(rad(i)), i)

    def test_sind(self):
        self.assertAlmostEqual(sind(30), .5)

    def test_cosd(self):
        self.assertAlmostEqual(cosd(60), .5)

    def test_tand(self):
        self.assertAlmostEqual(tand(30), 0.5773, 3)


    def test_mean_deg(self):
        self.assertEqual(mean_deg(np.array([0, 90])), 45)
        self.assertAlmostEqual(mean_deg(np.array([350, 10])), 0)


    def test_mean_deg_array(self):
        a = np.array([[0, 90], [350, 10], [0, -90]])
        np.testing.assert_array_almost_equal(mean_deg(a, 1), [45, 0, -45])
        np.testing.assert_array_almost_equal(mean_deg(a.T, 0), [45, 0, -45])

    def test_mean_deg_nan(self):
        self.assertEqual(mean_deg(np.array([0., 90, np.nan])), 45)


    def test_std_deg(self):
        self.assertEqual(std_deg(np.array([0, 0, 0])), 0)
        self.assertAlmostEqual(std_deg(np.array([0, 90, 180, 270])), 57.296, 2)

    def test_std_deg_nan(self):
        self.assertAlmostEqual(std_deg(np.array([0, 90, 180, 270, np.nan])), 57.296, 2)


    def test_rpm2rads(self):
        self.assertAlmostEqual(rpm2rads(1),.1047,4)
        self.assertAlmostEqual(rads2rpm(rpm2rads(1)), 1)
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_rad']
    unittest.main()
