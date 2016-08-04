# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 09:24:51 2016

@author: tlbl
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np


class Control(object):

    def pitch_controller_tuning(self, pitch, I, dQdt, P, Omr, om, csi):
        """

        Function to compute the gains of the pitch controller of the Basic DTU
        Wind Energy Controller with the pole placement technique implemented
        in HAWCStab2.

        Parameters
        ----------
        pitch: array
            Pitch angle [deg]
        I: array
            Drivetrain inertia [kg*m**2]
        dQdt: array
            Partial derivative of the aerodynamic torque with respect to the
            pitch angle [kNm/deg]. Can be computed with HAWCStab2.
        P: float
            Rated power [kW]. Set to zero in case of constant torque regulation
        Omr: float
            Rated rotational speed [rpm]
        om: float
            Freqeuncy of regulator mode [Hz]
        csi: float
            Damping ratio of regulator mode

        Returns
        -------
        kp: float
            Proportional gain [rad/(rad/s)]
        ki: float
            Intagral gain [rad/rad]
        K1: float
            Linear term of the gain scheduling [deg]
        K2: float
            Quadratic term of the gain shceduling [deg**2]


        """
        pitch = pitch * np.pi/180.
        I = I * 1e-3
        dQdt = dQdt * 180./np.pi
        Omr = Omr * np.pi/30.
        om = om * 2.*np.pi

        # Quadratic fitting of dQdt
        A = np.ones([dQdt.shape[0], 3])
        A[:, 0] = pitch**2
        A[:, 1] = pitch
        b = dQdt
        ATA = np.dot(A.T, A)
        iATA = np.linalg.inv(ATA)
        iATAA = np.dot(iATA, A.T)
        x = np.dot(iATAA, b)

        kp = -(2*csi*om*I[0] - P/(Omr**2))/x[2]
        ki = -(om**2*I[0])/x[2]

        K1 = x[2]/x[1]*(180./np.pi)
        K2 = x[2]/x[0]*(180./np.pi)**2

        return kp, ki, K1, K2


if __name__ == '__main__':

    pass
