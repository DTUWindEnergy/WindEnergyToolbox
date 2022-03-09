'''
Created on 05/06/2012

@author: Mads
'''
import os
from wetb.wind.utils import xyz2uvw
import wetb.gtsdf
from wetb.wind.shear import power_shear, fit_power_shear, fit_power_shear_ref, \
    log_shear, fit_log_shear, stability_term



import unittest
import numpy as np

import matplotlib.pyplot as plt


all = True
class TestShear(unittest.TestCase):

    def setUp(self):
        self.tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path

        """
        Sensor list of hdf5 files
        0: WSP gl. coo.,Abs_vhor 85
        1: WSP gl. coo.,Vdir_hor 85
        2: WSP gl. coo.,Abs_vhor 53
        3: WSP gl. coo.,Vdir_hor 53
        4: WSP gl. coo.,Abs_vhor 21
        5: WSP gl. coo.,Vdir_hor 21
        6: WSP gl. coo.,Vx 85
        7: WSP gl. coo.,Vy 85
        8: WSP gl. coo.,Vz 85
        9: WSP gl. coo.,Vx 53
        10: WSP gl. coo.,Vy 53
        11: WSP gl. coo.,Vz 53
        12: WSP gl. coo.,Vx 21
        13: WSP gl. coo.,Vy 21
        14: WSP gl. coo.,Vz 21
        """

    def test_power_shear_fit(self):
        z = [30,50,70]
        a,u_ref = .3,10
        u = power_shear(a,50,u_ref)(z)
        np.testing.assert_array_almost_equal(fit_power_shear_ref(zip(z,u), 50), [a,u_ref],3)
        if 0:
            import matplotlib.pyplot as plt
            fit_power_shear_ref(zip(z,u), 50, plt)
            plt.show()
        
    def test_power_shear_fit_nan(self):
        z = [30,50,70]
        a,u_ref = .3,10
        u = power_shear(a,50,u_ref)(z)
        u[2] = np.nan
        np.testing.assert_array_almost_equal(fit_power_shear_ref(zip(z,u), 50), [a,u_ref],3)
        u[:] = np.nan
        np.testing.assert_array_almost_equal(fit_power_shear_ref(zip(z,u), 50), [np.nan, np.nan],3)

    def test_power_shear(self):
        if all:
            _, data, _ = wetb.gtsdf.load(self.tfp + 'wind3.hdf5')
            u20 = data[:, 4]
            u70 = data[:, 6]
            z_u_lst1 = [(70, u70), (20, u20)]
            z_u_lst2 = [ (80, 6.93121004105), (100, 7.09558010101), (60, 6.58919000626), (40, 6.12744998932)]

            alpha = fit_power_shear(z_u_lst2)
            self.assertAlmostEqual(alpha, 0.163432280539)
            if 0:
                z = np.arange(10, 100)
                plt.plot([np.mean(u20), np.mean(u70)], [20, 70], 'bo')
                alpha = fit_power_shear(z_u_lst1)
                plt.plot(power_shear(alpha, 70, z, np.mean(u70)), z, 'b')

                z, u = np.array(z_u_lst2).T
                plt.plot(u, z, 'ro')
                alpha = fit_power_shear(z_u_lst2)
                z = np.sort(z)
                plt.plot(power_shear(alpha, 80, z, 6.9), z, 'r')
                plt.show()




    def test_fit_power_shear1(self):

        time, data, info = wetb.gtsdf.load(self.tfp + 'shear_noturb_85.hdf5')  #shear, alpha = 0.5

        wsp85 = data[:, 0]
        wsp53 = data[:, 2]
        wsp21 = data[:, 4]
        u85 = data[:, 7]
        u21 = data[:, 13]

        self.assertAlmostEqual(fit_power_shear([(85, wsp85), (53, wsp53), (21, wsp21)]), 0.5, delta=0.001)
        self.assertAlmostEqual(fit_power_shear([(85, wsp85), (21, wsp21)]), 0.5, delta=0.001)
        self.assertAlmostEqual(fit_power_shear([(85, u85), (21, u21)]), 0.5, delta=0.001)

    def test_fit_power_shear_ref(self):

        time, data, info = wetb.gtsdf.load(self.tfp + 'shear_noturb.hdf5')  #shear, alpha = 0.5

        wsp85 = data[:, 0]
        wsp53 = data[:, 2]
        wsp21 = data[:, 4]

        alpha, u_ref = fit_power_shear_ref([(85, wsp85), (21, wsp21)], 87.13333)
        self.assertAlmostEqual(alpha, .5, delta=.001)
        self.assertAlmostEqual(u_ref, 9, delta=.01)

        alpha, u_ref = fit_power_shear_ref([(85, wsp85), (53, wsp53), (21, wsp21)], 87.13333)
        self.assertAlmostEqual(alpha, .5, delta=.001)
        self.assertAlmostEqual(u_ref, 9, delta=.01)


    def test_fit_power_shear3(self):

        time, data, info = wetb.gtsdf.load(self.tfp + 'shear.hdf5')  #shear, alpha = 0.5
        #print "\n".join(["%d: %s" % (i, n) for i, n in enumerate(info['attribute_names'])])
        """
        0: WSP gl. coo.,Abs_vhor 85
        1: WSP gl. coo.,Vdir_hor 85
        2: WSP gl. coo.,Abs_vhor 53
        3: WSP gl. coo.,Vdir_hor 53
        4: WSP gl. coo.,Abs_vhor 21
        5: WSP gl. coo.,Vdir_hor 21
        6: WSP gl. coo.,Vx 85
        7: WSP gl. coo.,Vy 85
        8: WSP gl. coo.,Vz 85
        9: WSP gl. coo.,Vx 53
        10: WSP gl. coo.,Vy 53
        11: WSP gl. coo.,Vz 53
        12: WSP gl. coo.,Vx 21
        13: WSP gl. coo.,Vy 21
        14: WSP gl. coo.,Vz 21
        """
        wsp85 = data[:, 0]
        wsp53 = data[:, 2]
        wsp21 = data[:, 4]
        u85 = data[:, 7]
        u21 = data[:, 13]

        alpha, u_ref = fit_power_shear_ref([(85, wsp85), (21, wsp21)], 87.13333)
        self.assertAlmostEqual(alpha, .5, delta=.2)
        self.assertAlmostEqual(u_ref, 9, delta=.01)


    def test_log_shear(self):
        shear = log_shear(2, 3)
        self.assertAlmostEqual(shear(9), 5.49306144)

    def test_fit_log_shear(self):
        zu = [(85, 8.88131), (21, 4.41832)]
        u_star, z0 = fit_log_shear(zu)
        if 0:
            for z, u in zu:
                plt.plot(u, z, 'r.')
            z = np.arange(10, 100)
            plt.plot(log_shear(u_star, z0, z), z)
            plt.show()

        shear = log_shear(u_star, z0)
        for z,u in zu:
            self.assertAlmostEqual(u, shear(z), 4)


    def test_show_log_shear(self):
        if 0:
            for ustar in [1, 2]:
                for z0 in [1, 10]:
                    z = np.arange(z0, 200)
                    plt.plot(log_shear(ustar, z0, z), z, label="z0=%d, u*=%d" % (z0, ustar))
            plt.yscale('log')
            plt.legend()
            plt.show()

    def test_show_log_shear_stability(self):
        if 0:
            z0 = 1
            ustar = 1
            z = np.arange(z0, 200)
            for L in [-2000, -100, 100, 2000]:
                plt.plot(log_shear(ustar, z0, z, L), z, label="L=%d" % (L))
            #yscale('log')
            plt.legend()
            plt.show()



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
