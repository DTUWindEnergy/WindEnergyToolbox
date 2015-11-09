'''
Created on 16/07/2013

@author: mmpe
'''


import unittest
import numpy as np
from wetb.hawc2 import Hawc2io
from wetb.fatigue_tools.bearing_damage import bearing_damage_SWP


class TestBearingDamage(unittest.TestCase):

    def test_bearing_damage_swp(self):
        data = Hawc2io.ReadHawc2("test_files/test_bearing_damage").ReadBinary((np.array([4, 26, 6, 32, 8, 38]) - 1).tolist())
        self.assertAlmostEqual(bearing_damage_SWP([(data[:, i], data[:, i + 1]) for i in [0, 2, 4]]), 7.755595081475002e+13)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
