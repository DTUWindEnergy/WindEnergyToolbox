# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 09:24:51 2016

@author: tlbl
"""


import numpy as np


class Control(object):

    def torque_controller_tuning(self, r0, lambda_opt, ksi_pole_part,
                                 omega_pole_part, j_dt):
        """THIS IS A WIP
        """

        # TODO: THIS IS STILL A WIP

        i12, i23, i34 = self.select_regions()

        # Compute K factor for optimal CP tracking in region 1 as
        # K=0.5*rho*A*Cp_opt*R^3/lambda_opt^3

        # HAWCStab2/Computation_LIB/compute_controller_input_mod.f90:83-88

        # Compute K factor for optimal CP tracking in region 1 as
        # K=0.5*rho*A*Cp_opt*R^3/lambda_opt^3
        # pi_torque_controller%kfactor=0.d0
        # do i=1,iregion2-iregion1-1
        # !kfactor=kfactor+0.85d0*kfactor_fit(i)*substructure(3)
        #   pi_torque_controller%kfactor
        #   = pi_torque_controller%kfactor
        #   + kfactor_fit(i)*substructure(3)%opstate(i+iregion1)%r0**3/lambda_opt**3
        # enddo

        # HAWCStab2/Computation_LIB/compute_controller_input_mod.f90:89
        # pi_torque_controller%kfactor =
        #   pi_torque_controller%kfactor/max(dfloat(iregion2-iregion1-1), 1.d0)

        Qg = (r0*r0*r0)/(lambda_opt*lambda_opt*lambda_opt)

        # PI generator torque controller in region 2 (constant speed, variable power)
        # HAWCStab2/Computation_LIB/compute_controller_input_mod.f90:104-105
        # pi_torque_controller%kp = 2.d0*ksi_pole_part*omega_pole_part*j_dt
        # pi_torque_controller%ki = omega_pole_part**2*j_dt
        pgTorque = 2.0*ksi_pole_part*omega_pole_part*j_dt
        igTorque = omega_pole_part**2*j_dt

        return Qg, pgTorque, igTorque


    def pitch_controller_tuning(self, pitch, I, dQdt, P, Omr, om, csi):
        """

        Function to compute the gains of the pitch controller of the Basic DTU
        Wind Energy Controller with the pole placement technique implemented
        in HAWCStab2.

        Parameters
        ----------
        pitch: array
            Pitch angle [deg]. The values should only be those of the full load
            region.
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

    def K_omega2(V, P, R, TSR):

        Va = np.array(V)
        Pa = np.array(P)
        Ra = np.array(R)
        TSRa = np.array(TSR)
        K = Ra**3 * np.mean(Pa/(TSRa*Va)**3)

        return K

    def select_regions(self, pitch, omega, power):
        """Find indices at wich point the controller should switch between the
        different operating mode regions.

        Parameters
        ----------



        Returns
        ------

        i12 : int

        123 : int

        134 : int

        """

        i12 = 0

        n = len(pitch)

        for i in range(n-1):
            if (abs(power[i]/power[i+1] - 1.) > 0.01):
                if (abs(omega[i] / omega[i+1] - 1.) > 0.01):
                    i12 = i
                    break
        i23 = n-1
        for i in range(i12, n-1):
            if (abs(omega[i] / omega[i+1] - 1.) < 0.01):
                i23 = i
                break

        i34 = i23
        for i in range(i23, n-1):
            if (abs(power[i]/power[i+1] - 1.) > 0.01):
                if (abs(omega[i] / omega[i+1] - 1.) < 0.01):
                    i34 = i+1

        return i12, i23, i34


if __name__ == '__main__':

    pass
