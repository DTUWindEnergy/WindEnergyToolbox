'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import os
import tempfile

import numpy as np

from wetb.prepost import windIO


class TestsLoadResults(unittest.TestCase):

    def setUp(self):
        self.respath = os.path.join(os.path.dirname(__file__),
                                    '../../hawc2/tests/test_files/hawc2io/')
        self.fascii = 'Hawc2ascii'
        self.fbin = 'Hawc2bin'

    def loadresfile(self, resfile):
        res = windIO.LoadResults(self.respath, resfile)
        self.assertTrue(hasattr(res, 'sig'))
        self.assertEqual(res.Freq, 40.0)
        self.assertEqual(res.N, 800)
        self.assertEqual(res.Nch, 28)
        self.assertEqual(res.Time, 20.0)
        self.assertEqual(res.sig.shape, (800, 28))
        return res

    def test_load_ascii(self):
        res = self.loadresfile(self.fascii)
        self.assertEqual(res.FileType, 'ASCII')

    def test_load_binary(self):
        res = self.loadresfile(self.fbin)
        self.assertEqual(res.FileType, 'BINARY')

    def test_compare_ascii_bin(self):
        res_ascii = windIO.LoadResults(self.respath, self.fascii)
        res_bin = windIO.LoadResults(self.respath, self.fbin)

        for k in range(res_ascii.sig.shape[1]):
            np.testing.assert_allclose(res_ascii.sig[:,k], res_bin.sig[:,k],
                                       rtol=1e-02, atol=0.001)

    def test_unified_chan_names(self):
        res = windIO.LoadResults(self.respath, self.fascii, readdata=False)
        self.assertFalse(hasattr(res, 'sig'))

        np.testing.assert_array_equal(res.ch_df.index.values, np.arange(0,28))
        self.assertEqual(res.ch_df.unique_ch_name.values[0], 'Time')
        self.assertEqual(res.ch_df.unique_ch_name.values[27],
                         'windspeed-global-Vy--2.50-1.00--52.50')


class TestUserWind(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.z_h = 100.0
        self.r_blade_tip = 50.0
        self.h_ME = 650
        self.z = np.array([self.z_h - self.r_blade_tip,
                           self.z_h + self.r_blade_tip])

    def test_deltaphi2aphi(self):

        uwind = windIO.UserWind()
        profiles = windIO.WindProfiles

        for a_phi_ref in [-1.0, 0.0, 0.5]:
            phis = profiles.veer_ekman_mod(self.z, self.z_h, h_ME=self.h_ME,
                                           a_phi=a_phi_ref)
            d_phi_ref = phis[1] - phis[0]
#            a_phi1 = uwind.deltaphi2aphi(d_phi_ref, self.z_h, self.r_blade_tip,
#                                         h_ME=self.h_ME)
            a_phi2 = uwind.deltaphi2aphi_opt(d_phi_ref, self.z, self.z_h,
                                             self.r_blade_tip, self.h_ME)
            self.assertAlmostEqual(a_phi_ref, a_phi2)

    def test_usershear(self):

        uwind = windIO.UserWind()

        # phi, shear, wdir
        combinations = [[1,0,0], [0,-0.2,0], [0,0,-10], [None, None, None],
                        [0.5,0.2,10]]

        for a_phi, shear, wdir in combinations:
            rpl = (a_phi, shear, wdir)
            try:
                fname = 'a_phi_%1.05f_shear_%1.02f_wdir%02i.txt' % rpl
            except:
                fname = 'a_phi_%s_shear_%s_wdir%s.txt' % rpl
            target = os.path.join(self.path, fname)

            fid = tempfile.NamedTemporaryFile(delete=False, mode='wb')
            target = os.path.join(self.path, fname)
            uu, vv, ww, xx, zz = uwind(self.z_h, self.r_blade_tip, a_phi=a_phi,
                                       nr_vert=5, nr_hor=3, h_ME=650.0,
                                       wdir=wdir, io=fid, shear_exp=shear)
            # FIXME: this has to be done more clean and Pythonic
            # load again for comparison with the reference
            uwind.fid.close()
            with open(uwind.fid.name) as fid:
                contents = fid.readlines()
            os.remove(uwind.fid.name)
            with open(target) as fid:
                ref = fid.readlines()
            self.assertEqual(contents, ref)


if __name__ == "__main__":
    unittest.main()
