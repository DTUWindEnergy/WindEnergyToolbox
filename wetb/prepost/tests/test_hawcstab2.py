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
from os.path import join as pjoin
from os.path import dirname as pdirname

import numpy as np

from wetb.prepost.hawcstab2 import results, ReadControlTuning


class Tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fpath_linear = pjoin(pdirname(__file__),
                                  'data/controller_input_linear.txt')
        self.fpath_quad = pjoin(pdirname(__file__),
                                'data/controller_input_quadratic.txt')

    def test_cmb_df(self):
        fname1 = pjoin(pdirname(__file__), 'data/campbell_diagram.cmb')
        speed, freq, damp, real_eig = results().load_cmb(fname1)

        self.assertIsNone(real_eig)

        df = results().load_cmb_df(fname1)
        mods = freq.shape[1]
        ops = freq.shape[0]

        self.assertEqual(len(speed), ops)
        self.assertEqual(ops, 22)
        self.assertEqual(mods, 10)

        for k in range(ops):
            df_oper = df[df['wind_ms']==speed[k]]
            np.testing.assert_allclose(freq[k,:], df_oper['Fd_hz'].values)
            np.testing.assert_allclose(damp[k,:], df_oper['damp_ratio'].values)
            np.testing.assert_allclose(np.arange(1,len(df_oper)+1), df_oper['mode'])
            self.assertEqual(len(df_oper['wind_ms'].unique()), 1)
            self.assertEqual(df_oper['wind_ms'].unique()[0], speed[k])

    def test_linear_file(self):

        hs2 = ReadControlTuning()
        hs2.read_parameters(self.fpath_linear)

        self.assertEqual(hs2.pi_gen_reg1.K, 0.108313E+07)

        self.assertEqual(hs2.pi_gen_reg2.I, 0.307683E+08)
        self.assertEqual(hs2.pi_gen_reg2.Kp, 0.135326E+08)
        self.assertEqual(hs2.pi_gen_reg2.Ki, 0.303671E+07)

        self.assertEqual(hs2.pi_pitch_reg3.Kp, 0.276246E+01)
        self.assertEqual(hs2.pi_pitch_reg3.Ki, 0.132935E+01)
        self.assertEqual(hs2.pi_pitch_reg3.K1, 5.79377)
        self.assertEqual(hs2.pi_pitch_reg3.K2, 0.0)

        self.assertEqual(hs2.aero_damp.Kp2, 0.269403E+00)
        self.assertEqual(hs2.aero_damp.Ko1, -4.21472)
        self.assertEqual(hs2.aero_damp.Ko2, 0.0)

    def test_quadratic_file(self):

        hs2 = ReadControlTuning()
        hs2.read_parameters(self.fpath_quad)

        self.assertEqual(hs2.pi_gen_reg1.K, 0.108313E+07)

        self.assertEqual(hs2.pi_gen_reg2.I, 0.307683E+08)
        self.assertEqual(hs2.pi_gen_reg2.Kp, 0.135326E+08)
        self.assertEqual(hs2.pi_gen_reg2.Ki, 0.303671E+07)

        self.assertEqual(hs2.pi_pitch_reg3.Kp, 0.249619E+01)
        self.assertEqual(hs2.pi_pitch_reg3.Ki, 0.120122E+01)
        self.assertEqual(hs2.pi_pitch_reg3.K1, 7.30949)
        self.assertEqual(hs2.pi_pitch_reg3.K2, 1422.81187)

        self.assertEqual(hs2.aero_damp.Kp2, 0.240394E-01)
        self.assertEqual(hs2.aero_damp.Ko1, -1.69769)
        self.assertEqual(hs2.aero_damp.Ko2, -15.02688)

    def test_ind_file(self):
        fnames = ['dtu10mw_nofull_defl_u10000.ind',
                  'dtu10mw_nofull_fext_u10000.ind',
                  'dtu10mw_nofull_u10000.ind',
                  'dtu10mw_nogradient_defl_u10000.ind',
                  'dtu10mw_nogradient_fext_u10000.ind',
                  'dtu10mw_nogradient_u10000.ind',
                  'dtu10mw_v1_defl_u10000.ind',
                  'dtu10mw_v1_fext_u10000.ind',
                  'dtu10mw_v1_u10000.ind',
                  ]

        for fname in fnames:
            fname = pjoin(pdirname(__file__), 'data', fname)
            res = results()
            df_data = res.load_ind(fname)
            data = np.loadtxt(fname)
            np.testing.assert_almost_equal(data, df_data.values)

    def test_pwr_file(self):
        fnames = ['dtu10mw_nofull.pwr',
                  'dtu10mw_nogradient.pwr',
                  'dtu10mw_nogradient_v2.pwr',
                  'dtu10mw_v1.pwr',]
        for fname in fnames:
            fname = pjoin(pdirname(__file__), 'data', fname)
            res = results()
            df_data, units = res.load_pwr_df(fname)
            data = np.loadtxt(fname)
            self.assertEqual(data.shape, df_data.shape)
            np.testing.assert_almost_equal(data, df_data.values, decimal=2)


if __name__ == "__main__":
    unittest.main()
