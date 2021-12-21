'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
from os.path import join as pjoin
from os.path import dirname as pdirname

import numpy as np

from wetb.prepost.hawcstab2 import (results, ReadControlTuning, read_cmb_all,
                                    read_modid, PlotCampbell, plot_add_ps)
from wetb.prepost.mplutils import subplots


class Tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fbase = pdirname(__file__)
        self.fpath_linear = pjoin(self.fbase, 'data/controller_input_linear.txt')
        self.fpath_quad = pjoin(self.fbase, 'data/controller_input_quadratic.txt')

    def test_cmb_df(self):
        fname1 = pjoin(self.fbase, 'data/campbell_diagram.cmb')
        speed, freq, damp, real_eig = results().load_cmb(fname1)

        self.assertIsNone(real_eig)

        df = results().load_cmb_df(fname1)
        mods = freq.shape[1]
        ops = freq.shape[0]

        self.assertEqual(len(speed), ops)
        self.assertEqual(ops, 21)
        self.assertEqual(mods, 10)

        for k in range(ops):
            df_oper = df[df['wind_ms']==speed[k]]
            np.testing.assert_allclose(freq[k,:], df_oper['Fd_hz'].values)
            np.testing.assert_allclose(damp[k,:], df_oper['damp_ratio'].values)
            np.testing.assert_allclose(np.arange(1,len(df_oper)+1), df_oper['mode'])
            self.assertEqual(len(df_oper['wind_ms'].unique()), 1)
            self.assertEqual(df_oper['wind_ms'].unique()[0], speed[k])

    def test_read_cmb_all(self):

        f_pwr = pjoin(self.fbase, 'data/dtu10mw_v1.pwr')
        f_cmb = pjoin(self.fbase, 'data/campbell_diagram.cmb')
        f_modid = pjoin(self.fbase, 'data/dtu10mw.modid')
        dfp, dff, dfd = read_cmb_all(f_cmb, f_pwr=f_pwr, f_modid=f_modid)
        self.assertEqual(dfp.shape, (21, 27))
        self.assertEqual(dff.shape, (21, 10))
        self.assertEqual(dfd.shape, (21, 10))

        dfp, dff, dfd = read_cmb_all(f_cmb, f_pwr=None)
        self.assertIsNone(dfp)

    def test_read_modid(self):
        fname = pjoin(self.fbase, 'data/dtu10mw.modid')
        modes = read_modid(fname)
        ref = ['', '1st Tower FA', '1st Tower SS', '1st BF B whirling',
               '1st BF collective', '1st BF F whirling', '1st BE B whirling',
               '1st BE F whirling', '2nd BF B whirling', '2nd BF F whirling',
               '2nd BF collective', '1st shaft / BE collective',
               '2nd Tower FA', '2nd Tower SS', 'Tower torsion']
        self.assertEqual(len(modes), 25)
        self.assertEqual(modes[:15], ref)

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

        self.assertEqual(hs2.aero_gains.shape, (0, 0))

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

        self.assertEqual(hs2.aero_gains.shape, (15, 5))
        cols = ['theta', 'dq/dtheta', 'dq/dtheta_fit', 'dq/domega',
                'dq/domega_fit']
        self.assertEqual(hs2.aero_gains.columns.tolist(), cols)

        tmp = np.array([0, 4.1, 6.69, 8.62, 10.26, 11.74, 13.1, 14.38, 15.59,
                        16.76, 17.88, 18.97, 20.03, 21.05, 22.05])
        np.testing.assert_allclose(hs2.aero_gains['theta'].values, tmp)

        tmp = [-1165.0486, -1665.72575, -2012.86015, -2290.61883,
                     -2535.50152, -2757.11114, -2991.31463, -3213.58048,
                     -3428.46978, -3642.914, -3858.46084, -4075.53879,
                     -4295.293, -4524.66782, -4758.6268]
        np.testing.assert_allclose(hs2.aero_gains['dq/dtheta'].values, tmp)

        tmp = [-1182.80164, -1655.44826, -1998.12171, -2275.67536, -2526.42508,
               -2764.46364, -2993.03195, -3216.75546, -3435.9122, -3654.91116,
               -3871.07886, -4087.58722, -4303.93692, -4517.52214, -4732.06052]
        np.testing.assert_allclose(hs2.aero_gains['dq/dtheta_fit'].values, tmp)

        tmp = [-393.03157, -6919.03943, -13119.30826, -18911.31597,
               -24632.87239, -30186.31522, -36257.79933, -42410.9345,
               -48626.47812, -55070.40445, -61702.38984, -68581.71761,
               -75700.65394, -83045.36607, -90639.34883]
        np.testing.assert_allclose(hs2.aero_gains['dq/domega'].values, tmp)

        tmp = [-950.85937, -6544.84749, -12659.67192, -18515.75425,
               -24364.04365, -30329.6103, -36386.82912, -42591.10977,
               -48904.89826, -55424.76312, -62048.0563, -68852.77188,
               -75809.68369, -82820.10608, -89993.97031]
        np.testing.assert_allclose(hs2.aero_gains['dq/domega_fit'].values, tmp)

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
            fname = pjoin(self.fbase, 'data', fname)
            res = results()
            df_data = res.load_ind(fname)
            data = np.loadtxt(fname)
            np.testing.assert_allclose(data, df_data.values)

    def test_pwr_file(self):
        fnames = ['dtu10mw_nofull.pwr',
                  'dtu10mw_nogradient.pwr',
                  'dtu10mw_nogradient_v2.pwr',
                  'dtu10mw_v1.pwr',]
        for fname in fnames:
            fname = pjoin(self.fbase, 'data', fname)
            res = results()
            df_data, units = res.load_pwr_df(fname)
            data = np.loadtxt(fname)
            self.assertEqual(data.shape, df_data.shape)
            np.testing.assert_allclose(data, df_data.values)

    def test_opt_file(self):

        res = results()

        fname = pjoin(self.fbase, 'data', 'dtu10mw.opt')
        df = res.load_operation(fname)
        tmp = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25]
        np.testing.assert_allclose(tmp, df['windspeed'].values)
        self.assertEqual(df.values.shape, (21, 3))

        fname = pjoin(pdirname(__file__), 'data', 'kb6.opt')
        df = res.load_operation(fname)
        tmp = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25]
        np.testing.assert_allclose(tmp, df['windspeed'].values)
        tmp = [7.20147212792062, 14.1560777401151, 24.5923689569214,
               39.2304595043255, 58.7985344647829, 83.2307972895692,
               106.401798353616, 106.400099017382, 106.403429471703,
               106.3986284106, 106.39861469995, 106.393720086642,
               106.404364646693, 106.401389916882, 106.4047949236,
               106.398523413816, 106.403149136568, 106.399813867904,
               106.424042832599, 106.400584861663, 106.43349513828,
               106.469433544515]
        np.testing.assert_allclose(tmp, df['P_aero'].values)
        self.assertEqual(df.values.shape, (22, 5))

    def test_plot_cmb(self):

        base = pdirname(__file__)
        f_pwr = pjoin(base, 'data/dtu10mw_v1.pwr')
        f_cmb = pjoin(base, 'data/campbell_diagram.cmb')
        dfp, dff, dfd = read_cmb_all(f_cmb, f_pwr=f_pwr)
        cmb = PlotCampbell(dfp['V'].values, dff, dfd)

        fig, axes = subplots(nrows=2, ncols=1, figsize=(8,10))
        ax = axes[0,0]
        ax = cmb.plot_freq(ax, col='k', mark='^', ls='-', modes='all')
        ax = plot_add_ps(ax, dfp['V'], dfp['Speed'], ps=[1,3,6])
        ax = axes[1,0]
        ax = cmb.plot_damp(ax, col='k', mark='^', ls='-', modes=10)


if __name__ == "__main__":
    unittest.main()
