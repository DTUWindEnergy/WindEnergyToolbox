# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:12:58 2014

@author: dave
"""



import unittest
import os
import re

import numpy as np
import pandas as pd

from wetb.prepost import mplutils


class dummy:
    def __init__(self):
        pass


def ReadFileHAWCStab2Header(fname, widths):
    """
    Read a file with a weird HAWCStab2 header that starts with a #, and
    includes the column number and units between square brackets.
    """

    regex = re.compile('(\\[.*?\\])')

    def _newformat(fname):
        df = pd.read_fwf(fname, header=0, widths=[20]*15)
        # find all units
        units = regex.findall(''.join(df.columns))
        df.columns = [k[:-2].replace('#', '').strip() for k in df.columns]
        return df, units

    def _oldformat(fname):
        df = pd.read_fwf(fname, header=0, widths=[14]*13)
        # find all units
        units = regex.findall(''.join(df.columns))
        df.columns = [k.replace('#', '').strip() for k in df.columns]
        return df, units

    with open(fname) as f:
        line = f.readline()

    if len(line) > 200:
        return _newformat(fname)
    else:
        return _oldformat(fname)


class InductionResults(object):
    def __init__(self):
        pass
    def read(self, fname):
        self.data = np.loadtxt(fname)
        self.wsp = int(fname.split('_u')[-1][:-4]) / 1000.0
        try:
            self.df_data = pd.read_fwf(fname, header=0, widths=[14]*38)
        except:
            self.df_data = pd.read_fwf(fname, header=0, widths=[14]*34)
        # sanitize the headers
        cols = self.df_data.columns
        self.df_data.columns = [k[:-2].replace('#', '').strip() for k in cols]


class results(object):
    """
    Loading HAWCStab2 result files
    """

    def __init__(self):
        pass

    def load_pwr(self, fname):
        pwr = np.loadtxt(fname)

        res = dummy()

        res.wind = pwr[:,0]
        res.power = pwr[:,1]
        res.thrust = pwr[:,2]
        res.cp = pwr[:,3]
        res.ct = pwr[:,4]
        res.pitch_deg = pwr[:,8]
        res.rpm = pwr[:,9]

        return res

    def load_pwr_df(self, fname):
        return ReadFileHAWCStab2Header(fname, [20]*15)

    def load_cmb(self, fname):
        cmb = np.loadtxt(fname)
        # when there is only data for one operating condition we only have one
        # row and consequently only a 1D array
        if len(cmb.shape) == 1:
            cmb = cmb.reshape( (1, cmb.shape[0]) )
        wind = cmb[:,0]
        ii = int((cmb.shape[1]-1)/2)
        freq = cmb[:,1:ii+1]
        damp = cmb[:,ii+1:]

        return wind, freq, damp

    def load_cmb_df(self, fname):
        # index name can be rotor speed or wind speed
        with open(fname) as f:
            header = f.readline()
        oper_name = header.split('1')[0].strip().replace('#', '').lower()
        oper_name = oper_name.replace(' ', '').replace('[', '_')[:-1]
        oper_name = oper_name.replace('/', '')

        speed, freq, damp = self.load_cmb(fname)
        mods = freq.shape[1]
        ops = freq.shape[0]

        df = pd.DataFrame(columns=[oper_name, 'Fd_hz', 'damp_ratio', 'mode'])
        df['Fd_hz'] = freq.flatten()
        df['damp_ratio'] = damp.flatten()
        # now each mode number is a row so that means that each operating
        # point is now repeated as many times as there are modes
        df[oper_name] = speed.repeat(mods)
        modes = np.arange(1, mods+1, 1)
        df['mode'] = modes.reshape((1,mods)).repeat(ops, axis=0).flatten()

        return df

    def load_frf(self, fname, nr_inputs=3):
        frf = np.loadtxt(fname)

        self.nr_outputs = ((frf.shape[1] - 1) / 2) / nr_inputs
        self.nr_inputs = nr_inputs

        return frf

    def load_ind(self, fname):
        self.ind = InductionResults()
        self.ind.read(fname)

    def load_operation(self, fname):

        operation = np.loadtxt(fname, skiprows=1)
        # when the array is empty, set operation to an empty DataFrame
        if len(operation) == 0:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm']
            self.operation = pd.DataFrame(columns=cols)
            return
        # when there is only one data point, the array is 1D, we allways need
        # a 2D array otherwise the columns become rows in the DataFrame
        elif len(operation.shape) == 1:
            operation = operation.reshape((1, operation.shape[0]))
        try:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm']
            self.operation = pd.DataFrame(operation, columns=cols)
        except ValueError:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm', 'P_aero',
                    'T_aero']
            self.operation = pd.DataFrame(operation, columns=cols)

    def write_ae_sections_h2(self):
        """
        Get the aerosection positions from the HS2 ind result file and
        write them as outputs for HAWC2
        """
        self.ind

    def plot_pwr(self, figname, fnames, labels=[], figsize=(11,7.15), dpi=120):

        results = []
        if isinstance(fnames, list):
            if len(fnames) > 4:
                raise ValueError('compare up to maximum 4 HawcStab2 cases')
            for fname in fnames:
                results.append(self.load_pwr(fname))
                # if the labels are not defined, take the file name
                if len(labels) < len(fnames):
                    labels.append(os.path.basename(fname))
        else:
            results.append(self.load_pwr(fname))

        colors = list('krbg')
        symbols = list('o<+x')
        alphas = [1.0, 0.9, 0.8, 0.75]

        fig, axes = mplutils.subplots(nrows=2, ncols=2, figsize=figsize,
                                       dpi=dpi, num=0)
        for i, res in enumerate(results):
            ax = axes[0,0]
            ax.plot(res.wind, res.power, color=colors[i], label=labels[i],
                    marker=symbols[i], ls='-', alpha=alphas[i])
            ax.set_title('Aerodynamic Power [kW]')

            ax = axes[0,1]
            ax.plot(res.wind, res.pitch_deg, color=colors[i], label=labels[i],
                    marker=symbols[i], ls='-', alpha=alphas[i])
            ax.set_title('Pitch [deg]')

            ax = axes[1,0]
            ax.plot(res.wind, res.thrust, color=colors[i], label=labels[i],
                    marker=symbols[i], ls='-', alpha=alphas[i])
            ax.set_title('Thrust [kN]')

            ax = axes[1,1]
            ax.plot(res.wind, res.cp, label='$C_p$ %s ' % labels[i], ls='-',
                    color=colors[i], marker=symbols[i], alpha=alphas[i])
            ax.plot(res.wind, res.ct, label='$C_t$ %s ' % labels[i], ls='--',
                    color=colors[i], marker=symbols[i], alpha=alphas[i])
            ax.set_title('Power and Thrust coefficients [-]')

        for ax in axes.ravel():
            ax.legend(loc='best')
            ax.grid(True)
            ax.set_xlim([res.wind[0], res.wind[-1]])
        fig.tight_layout()

        print('saving figure: %s ... ' % figname, end='')
        figpath = os.path.dirname(figname)
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        fig.savefig(figname)
        fig.clear()
        print('done!')


class hs2_control_tuning(object):

    def __init__(self):
        """
        """
        pass

    def parse_line(self, line, controller):

        split1 = line.split('=')
        var1 = split1[0].strip()
        try:
            val1 = float(split1[1].split('[')[0])
            attr = getattr(self, controller)
            setattr(attr, var1, val1)

            if len(split1) > 2:
                var2 = split1[1].split(',')[1].strip()
                val2 = float(split1[2].split('[')[0])
                setattr(attr, var2, val2)
        except IndexError:
            pass

    def read_parameters(self, fpath):
        """
        Read the controller tuning file
        ===============================

        """

        with open(fpath, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    controller = 'pi_gen_reg1'
                    setattr(self, controller, dummy())
                elif i == 2:
                    controller = 'pi_gen_reg2'
                    setattr(self, controller, dummy())
                elif i == 6:
                    controller = 'pi_pitch_reg3'
                    setattr(self, controller, dummy())
                elif i == 10:
                    controller = 'aero_damp'
                    setattr(self, controller, dummy())
                else:
                    self.parse_line(line, controller)

        # set some parameters to zero for the linear case
        if not hasattr(self.pi_pitch_reg3, 'K2'):
            setattr(self.pi_pitch_reg3, 'K2', 0.0)
        if not hasattr(self.aero_damp, 'Ko2'):
            setattr(self.aero_damp, 'Ko2', 0.0)


class tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fpath_linear = 'data/controller_input_linear.txt'
        self.fpath_quadratic = 'data/controller_input_quadratic.txt'

    def test_cmb_df(self):
        fname1 = 'data/campbell_wind.cmb'
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


if __name__ == '__main__':

    unittest.main()
