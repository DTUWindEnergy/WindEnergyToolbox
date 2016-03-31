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

import numpy as np

from wetb.prepost.hawcstab2 import results, hs2_control_tuning


# path for test data files
#fpath = os.path.join(os.path.dirname(__file__), 'data/')

class Tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fpath_linear = 'data/controller_input_linear.txt'
        self.fpath_quadratic = 'data/controller_input_quadratic.txt'

    def test_cmb_df(self):
        fname1 = 'data/campbell_diagram.cmb'
        speed, freq, damp = results().load_cmb(fname1)

        df = results().load_cmb_df(fname1)
        #mods = freq.shape[1]
        ops = freq.shape[0]

        self.assertEqual(len(speed), ops)

        for k in range(ops):
            df_oper = df[df['wind_ms']==speed[k]]
            np.testing.assert_allclose(freq[k,:], df_oper['Fd_hz'].values)
            np.testing.assert_allclose(damp[k,:], df_oper['damp_ratio'].values)
            np.testing.assert_allclose(np.arange(1,len(df_oper)+1), df_oper['mode'])
            self.assertEqual(len(df_oper['wind_ms'].unique()), 1)
            self.assertEqual(df_oper['wind_ms'].unique()[0], speed[k])

    def test_linear_file(self):

        hs2 = hs2_control_tuning()
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

        hs2 = hs2_control_tuning()
        hs2.read_parameters(self.fpath_quadratic)

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


if __name__ == "__main__":
    unittest.main()
