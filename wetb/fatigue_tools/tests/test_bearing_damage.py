'''
Created on 16/07/2013

@author: mmpe
'''
from wetb import gtsdf


import unittest

import numpy as np
from wetb.hawc2 import Hawc2io
from wetb.fatigue_tools.bearing_damage import bearing_damage
import os


class TestBearingDamage(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tfp = os.path.join(os.path.dirname(__file__), 'test_files/')

    def test_bearing_damage_swp(self):
        data = Hawc2io.ReadHawc2(self.tfp + "test_bearing_damage").ReadBinary((np.array([1, 4, 2, 5, 3, 6]) ).tolist())
        self.assertAlmostEqual(bearing_damage([(data[:, i], data[:, i + 1]) for i in [0,2,4]]), 7.755595081475002e+13)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
