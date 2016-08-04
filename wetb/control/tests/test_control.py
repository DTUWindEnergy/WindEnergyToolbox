# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 11:09:43 2016

@author: tlbl
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

from wetb.control import control
import numpy as np


class TestControl(unittest.TestCase):

    def setUp(self):
        dQdt = np.array([-0.126876918E+03, -0.160463547E+03, -0.211699586E+03,
                         -0.277209984E+03, -0.351476131E+03, -0.409411354E+03,
                         -0.427060299E+03, -0.608498644E+03, -0.719141594E+03,
                         -0.814129630E+03, -0.899248007E+03, -0.981457622E+03,
                         -0.106667910E+04, -0.114961807E+04, -0.123323210E+04,
                         -0.131169210E+04, -0.138758236E+04, -0.145930419E+04,
                         -0.153029102E+04, -0.159737975E+04, -0.166846850E+04])

        pitch = np.array([0.2510780000E+00, 0.5350000000E-03, 0.535000000E-03,
                          0.5350000000E-03, 0.5350000000E-03, 0.535000000E-03,
                          0.5350000000E-03, 0.3751976000E+01, 0.625255700E+01,
                          0.8195032000E+01, 0.9857780000E+01, 0.113476710E+02,
                          0.1271615400E+02, 0.1399768300E+02, 0.152324310E+02,
                          0.1642177100E+02, 0.1755302300E+02, 0.186442750E+02,
                          0.1970333100E+02, 0.2073358600E+02, 0.217410280E+02])

        I = np.array([0.4394996114E+08, 0.4395272885E+08, 0.4395488725E+08,
                      0.4395301987E+08, 0.4394561932E+08, 0.4393327166E+08,
                      0.4391779133E+08, 0.4394706335E+08, 0.4395826989E+08,
                      0.4396263773E+08, 0.4396412693E+08, 0.4396397777E+08,
                      0.4396275304E+08, 0.4396076315E+08, 0.4395824699E+08,
                      0.4395531228E+08, 0.4395201145E+08, 0.4394837798E+08,
                      0.4394456127E+08, 0.4394060604E+08, 0.4393647769E+08])

        self.dQdt = dQdt
        self.pitch = pitch
        self.I = I

    def test_pitch_controller_tuning(self):

        crt = control.Control()
        P = 0.
        Omr = 12.1
        om = 0.10
        csi = 0.7
        i = 5
        kp, ki, K1, K2 = crt.pitch_controller_tuning(self.pitch[i:],
                                                     self.I[i:],
                                                     self.dQdt[i:],
                                                     P, Omr, om, csi)

        self.assertEqual(kp, 1.596090243644432)
        self.assertEqual(ki, 0.71632362627138424)
        self.assertEqual(K1, 10.01111637532056)
        self.assertEqual(K2, 599.53659803157643)

if __name__ == "__main__":

    unittest.main()
