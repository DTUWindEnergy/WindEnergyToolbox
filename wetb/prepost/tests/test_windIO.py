'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
import io
import os
import tempfile

import numpy as np
import pandas as pd

from wetb.prepost import windIO


class TestsLogFile(unittest.TestCase):

    def setUp(self):
        self.logpath = os.path.join(os.path.dirname(__file__),
                                    '../../hawc2/tests/test_files/logfiles/')

    def readlog(self, fname):
        log = windIO.LogFile()
        log.readlog(os.path.join(self.logpath, fname))
        return log

    def test_reading(self):
        fname = 'simulating.log'
        log = self.readlog(fname)
        self.assertTrue(hasattr(log, 'MsgListLog'))
        self.assertTrue(hasattr(log, 'MsgListLog2'))
        fpath = os.path.join(self.logpath, fname)
        self.assertEqual(len(log.MsgListLog), 1)
        self.assertEqual(len(log.MsgListLog2), 1)
        self.assertEqual(log.MsgListLog[0][0], fpath)
        self.assertTrue(fpath in log.MsgListLog2)
        # the current log file doesn't contain any errors but didn't complete
        # [found_error, exit_correct]
        self.assertEqual(log.MsgListLog2[fpath], [False, False])

    def test_loganalysis_file(self):
        fname = 'simulating.log'
        log = self.readlog(fname)
        csv = log._header()
        csv = log._msglistlog2csv(csv)
        # because our API is really crappy, we emulate writing to StringIO
        # instead of to a file
        fcsv = io.StringIO(csv)
        df = log.csv2df(fcsv)
        self.assertEqual(df.loc[0,'nr_time_steps'], 25)
        self.assertEqual(df.loc[0,'total_iterations'], 49)
        self.assertEqual(df.loc[0,'file_name'], log.MsgListLog[0][0])
        self.assertAlmostEqual(df.loc[0,'last_time_step'], 0.5, places=5)
        self.assertAlmostEqual(df.loc[0,'dt'], 0.02)
        self.assertAlmostEqual(df.loc[0,'max_iters_p_time_step'], 2.0)
        self.assertAlmostEqual(df.loc[0,'mean_iters_p_time_step'], 1.96)
        self.assertTrue(np.isnan(df.loc[0,'seconds_p_iteration']))

    def test_read_and_analysis_simulation_error2(self):

        fname = 'simulation_error2.log'
        fpath = os.path.join(self.logpath, fname)

        log = self.readlog(fname)
        # finish correctly, but with errors, [found_error, exit_correct]
        self.assertEqual(log.MsgListLog2[fpath], [True, True])

        csv = log._header()
        csv = log._msglistlog2csv(csv)
        # because our API is really crappy, we emulate writing to StringIO
        # instead of to a file
        fcsv = io.StringIO(csv)
        df = log.csv2df(fcsv)
        self.assertEqual(df.loc[0,'nr_time_steps'], 1388)
        self.assertEqual(df.loc[0,'total_iterations'], 0)
        self.assertEqual(df.loc[0,'file_name'], log.MsgListLog[0][0])
        self.assertAlmostEqual(df.loc[0,'dt'], 0.02)
        self.assertAlmostEqual(df.loc[0,'max_iters_p_time_step'], 0.0)
        self.assertAlmostEqual(df.loc[0,'mean_iters_p_time_step'], 0.0)
        self.assertAlmostEqual(df.loc[0,'elapsted_time'], 0.3656563)
        self.assertAlmostEqual(df.loc[0,'last_time_step'], 27.76, places=5)
        self.assertAlmostEqual(df.loc[0,'real_sim_time'], 75.9183, places=4)
        self.assertTrue(np.isnan(df.loc[0,'seconds_p_iteration']))

        self.assertEqual(df.loc[0,'first_tstep_104'], 1386)
        self.assertEqual(df.loc[0,'last_step_104'], 1388)
        self.assertEqual(df.loc[0,'nr_104'], 30)
        msg = ' *** ERROR *** Out of limits in user defined shear field - '
        msg += 'limit value used'
        self.assertEqual(df.loc[0,'msg_104'], msg)

    def test_read_and_analysis_init_error(self):

        fname = 'init_error.log'
        fpath = os.path.join(self.logpath, fname)

        log = self.readlog(fname)
        # errors, but no sim time or finish message [found_error, exit_correct]
        self.assertEqual(log.MsgListLog2[fpath], [True, True])

        csv = log._header()
        csv = log._msglistlog2csv(csv)
        # because our API is really crappy, we emulate writing to StringIO
        # instead of to a file
        fcsv = io.StringIO(csv)
        df = log.csv2df(fcsv)

        msg = ' *** ERROR *** No line termination in command line            8'
        self.assertEqual(df.loc[0,'msg_5'], msg)

    def test_read_and_analysis_init(self):

        fname = 'init.log'
        fpath = os.path.join(self.logpath, fname)

        log = self.readlog(fname)
        # errors, but no sim time or finish message [found_error, exit_correct]
        self.assertEqual(log.MsgListLog2[fpath], [False, False])

        csv = log._header()
        csv = log._msglistlog2csv(csv)
        # because our API is really crappy, we emulate writing to StringIO
        # instead of to a file
        fcsv = io.StringIO(csv)
        df = log.csv2df(fcsv)

    def test_read_and_analysis_tmp(self):

        fname = 'tmp.log'
        fpath = os.path.join(self.logpath, fname)

        log = self.readlog(fname)
        csv = log._header()
        csv = log._msglistlog2csv(csv)
        fcsv = io.StringIO(csv)
        df = log.csv2df(fcsv)
        # finish correctly, but with errors
        self.assertAlmostEqual(df.loc[0,'elapsted_time'], 291.6350, places=5)
        self.assertEqual(log.MsgListLog2[fpath], [True, True])


class TestsLoadResults(unittest.TestCase):

    def setUp(self):
        self.respath = os.path.join(os.path.dirname(__file__),
                                    '../../hawc2/tests/test_files/hawc2io/')
        self.fascii = 'Hawc2ascii'
        self.fbin = 'Hawc2bin'
        self.f1_chant = 'hawc2ascii_chantest_1.sel'
        self.f2_chant = 'hawc2bin_chantest_2.sel'
        self.f3_chant = 'hawc2bin_chantest_3.sel'

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

    def test_unified_chan_names_extensive(self):

        # ---------------------------------------------------------------------
        res = windIO.LoadResults(self.respath, self.f1_chant, readdata=False)
        self.assertFalse(hasattr(res, 'sig'))
        np.testing.assert_array_equal(res.ch_df.index.values, np.arange(0,432))
        self.assertEqual(res.ch_df.unique_ch_name.values[0], 'Time')
        df = res.ch_df
        self.assertEqual(2, len(df[df['bearing_name']=='shaft_rot']))
        self.assertEqual(18, len(df[df['sensortype']=='State pos']))
        self.assertEqual(11, len(df[df['blade_nr']==1]))

        exp = [[38, 'global-blade2-elem-019-zrel-1.00-State pos-z', 'm'],
               [200, 'blade2-blade2-node-017-momentvec-z', 'kNm'],
               [296, 'blade1-blade1-node-008-forcevec-z', 'kN'],
               [415, 'Cl-1-55.7', 'deg'],
               [421, 'qwerty-is-azerty', 'is'],
               [422, 'wind_wake-wake_pos_x_1', 'm'],
               [423, 'wind_wake-wake_pos_y_2', 'm'],
               [424, 'wind_wake-wake_pos_z_5', 'm'],
               [425, 'statevec_new-blade1-c2def-blade1-absolute-014.00-Dx', 'm'],
               [429, 'statevec_new-blade1-c2def-blade1-elastic-014.00-Ry', 'deg'],
              ]
        for k in exp:
            self.assertEqual(df.loc[k[0], 'unique_ch_name'], k[1])
            self.assertEqual(df.loc[k[0], 'units'], k[2])
            self.assertEqual(res.ch_dict[k[1]]['chi'], k[0])
            self.assertEqual(res.ch_dict[k[1]]['units'], k[2])

        # also check we have the tag from a very long description because
        # we truncate after 150 characters
        self.assertEqual(df.loc[426, 'sensortag'], 'this is a tag')

        # ---------------------------------------------------------------------
        res = windIO.LoadResults(self.respath, self.f2_chant, readdata=False)
        self.assertFalse(hasattr(res, 'sig'))
        np.testing.assert_array_equal(res.ch_df.index.values, np.arange(0,217))
        df = res.ch_df
        self.assertEqual(13, len(df[df['sensortype']=='wsp-global']))
        self.assertEqual(3, len(df[df['sensortype']=='wsp-blade']))
        self.assertEqual(2, len(df[df['sensortype']=='harmonic']))
        self.assertEqual(2, len(df[df['blade_nr']==3]))

        # ---------------------------------------------------------------------
        res = windIO.LoadResults(self.respath, self.f3_chant, readdata=False)
        self.assertFalse(hasattr(res, 'sig'))
        np.testing.assert_array_equal(res.ch_df.index.values, np.arange(0,294))
        df1 = res.ch_df
        self.assertEqual(8, len(df1[df1['sensortype']=='CT']))
        self.assertEqual(8, len(df1[df1['sensortype']=='CQ']))
        self.assertEqual(8, len(df1[df1['sensortype']=='a_grid']))
        self.assertEqual(84, len(df1[df1['blade_nr']==1]))

    def test_unified_chan_names_extensive2(self):

        res = windIO.LoadResults(self.respath, self.f3_chant, readdata=False)
        df1 = res.ch_df

        fname = os.path.join(self.respath, self.f3_chant.replace('.sel',
                                                                 '.ch_df.csv'))
        # # when changing the tests, update the reference, check, and commit
        # # but keep the same column ordering to not make the diffs to big
        # cols = ['azimuth', 'bearing_name', 'blade_nr', 'bodyname',
        #         'component', 'coord', 'direction', 'dll', 'flap_nr', 'io',
        #         'io_nr', 'output_type', 'pos', 'radius','radius_actual', 'sensortag',
        #         'sensortype', 'unique_ch_name', 'units', ]
        # df1[cols].to_csv(fname)
        # df1.to_excel(fname.replace('.csv', '.xlsx'))

        # FIXME: read_csv for older pandas versions fails on reading the
        # mixed str/tuple column. Ignore the pos column for now
        colref = ['azimuth', 'bearing_name', 'blade_nr', 'bodyname',
                  'component', 'coord', 'direction', 'dll', 'flap_nr', 'io',
                  'io_nr', 'output_type', 'radius', 'radius_actual', 'sensortag',
                  'sensortype', 'unique_ch_name', 'units'] # 'pos',
        # keep_default_na: leave empyt strings as empty strings and not nan's
        # you can't have nice things: usecols in combination with index_col
        # doesn't work
        df2 = pd.read_csv(fname, usecols=['chi']+colref, keep_default_na=False)
        df2.index = df2['chi']
        df2.drop(labels='chi', inplace=True, axis=1)

        # for the comparison we need to have the columns with empty/number
        # mixed data types in a consistent data type
        for col in ['azimuth', 'radius', 'blade_nr', 'io_nr', 'flap_nr',
                    'dll', 'radius_actual']:
            df1.loc[df1[col]=='', col] = np.nan
            df1[col] = df1[col].astype(np.float32)
            df2.loc[df2[col]=='', col] = np.nan
            df2[col] = df2[col].astype(np.float32)

        # print(df1.pos[14], df2.pos[14])
        # the pos columns contains also tuples, read from csv doesn't get that
        # df1['pos'] = df1['pos'].astype(np.str)
        # df1['pos'] = df1['pos'].str.replace("'", "")
        # print(df1.pos[14], df2.pos[14])

        # sort columns in the same way so we can assert the df are equal
        # df1 = df1[colref].copy()
        # df2 = df2[colref].copy()
        # FIXME: when pandas is more recent we can use assert_frame_equal
        # pd.testing.assert_frame_equal(df1, df2)
        # ...but there is no testing.assert_frame_equal in pandas 0.14
        for col in colref:
            # ignore nan values for float cols
            if df1[col].dtype == np.dtype('float32'):
                sel = ~np.isnan(df1[col].values)
                np.testing.assert_array_equal(df1[col].values[sel],
                                              df2[col].values[sel])
            else:
                np.testing.assert_array_equal(df1[col].values,
                                              df2[col].values)

    def test_df_stats(self):
        """
        """
        res = windIO.LoadResults(self.respath, self.fascii)
        df_stats = res.statsdel_df()


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
