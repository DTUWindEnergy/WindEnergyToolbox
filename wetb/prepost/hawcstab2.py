"""
Created on Tue Jan 14 14:12:58 2014

@author: dave
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from io import open
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import object

import os
import re

import numpy as np
import pandas as pd

from wetb.prepost import (mplutils, misc)


class dummy(object):
    def __init__(self, name='dummy'):
        self.__name__ = name

regex_units = re.compile('(\\[.*?\\])')


def ReadFileHAWCStab2Header(fname):
    """
    Read a file with a weird HAWCStab2 header that starts with a #, and
    includes the column number and units between square brackets.
    """

    def get_lines(fname):
        # get the line that contains the header/column names and the first
        # line that holds the data
        with open(fname) as f:
            line_header = f.readline()
            line_data = f.readline()
            # sometimes there are more header lines. The header is always on the
            # last of the lines marked with #
            while line_data[:2].strip() == '#':
                line_header = line_data
                line_data = f.readline()
        return line_header, line_data

    def get_col_widths(line):
        # it is very annoying that various files can have various column widths
        # also, the first column is one character wider than the rest
        i0 = re.search(r'\S',line).start()
        i1_col1 = line[i0:].find(' ') + i0
        # number of columns can also be different (gradients or not, node displ)
        nr_cols = int(round(len(line)/i1_col1, 0))
        colwidths = [i1_col1+1] + [i1_col1]*(nr_cols-1)
        return colwidths

    def get_col_names(line, colwidths):
        # because sometimes there are no spaces between the header of each column
        # sanitize the headers
        ci = np.array([0] + colwidths).cumsum()
        # remember zero based indexing
        ci[1:] = ci[1:] - 1
        columns = []
        for i in range(len(ci)-1):
            # also lose the index in the header
            colname = line[ci[i]:ci[i+1]][:-2].replace('#', '').strip()
            columns.append(colname)
        return columns

    line_header, line_data = get_lines(fname)
    colwidths = get_col_widths(line_data)
    columns = get_col_names(line_header, colwidths)
    # gradients have duplicate columns: set for with wake updated
    # and another with frozen wake assumption, append _fw to the columns
    # used to indicate frozen wake gradients
    if 'dQ/dt [kNm/deg]' in columns:
        i1 = columns.index('dQ/dt [kNm/deg]')
        if i1 > -1:
            i2 = columns.index('dQ/dt [kNm/deg]', i1+1)
        if i2 > i1:
            for i in range(i2, len(columns)):
                columns[i] = columns[i].replace(' [', '_fw [')

    df = pd.read_fwf(fname, widths=colwidths, comment='#', header=None,
                     names=columns)
    units = regex_units.findall(''.join(columns))

    return df, units


class InductionResults(object):
    """Column width can vary between versions and with/withouth gradient in
    output. Use get_col_width() for automatic detection.
    """
    def __init__(self, colwidth=None):
        """with gradients currently ind has columns width of 28 instead of 14!
        """
        self.cw = colwidth

    def get_col_width(self, fname):
        # figure out column width
        with open(fname) as fid:
            # line1 contains the header
            line1 = fid.readline()
            # line2 contains the numerical data
            line2 = fid.readline()

        # it is very annoying that various files can have various column widths
        # also, the first column is one character wider than the rest
        i0 = re.search(r'\S',line2).start()
        self.i1_col1 = line2[i0:].find(' ') + i0
        # number of columns can also be different (gradients or not, node displ)
        nr_cols = int(round(len(line2)/self.i1_col1, 0))
        self.colwidths = [self.i1_col1+1] + [self.i1_col1]*(nr_cols-1)

        # because sometimes there are no spaces between the header of each column
        # sanitize the headers
        ci = np.array([0] + self.colwidths).cumsum()
        # remember zero based indexing
        ci[1:] = ci[1:] - 1
        self.columns = []
        for i in range(len(ci)-1):
            # also lose the index in the header
            colname = line1[ci[i]:ci[i+1]][:-2].replace('#', '').strip()
            self.columns.append(colname)

    def read(self, fname):
        if self.cw is None:
            self.get_col_width(fname)
        self.wsp = int(fname.split('_u')[-1][:-4]) / 1000.0
        # self.df_data = pd.read_fwf(fname, header=0, widths=self.colwidths)
        self.df_data = pd.read_fwf(fname, skiprows=0, widths=self.colwidths)
        # we can not rely on read_fwf for the column names since for some
        # columns there is no space between one column and the next one.
        self.df_data.columns = self.columns


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
        return ReadFileHAWCStab2Header(fname)

    def load_cmb(self, fname):
        # aero-(servo)-elastic results for HS2>=2.14 have real_eig as 3th set
        with open(fname) as f:
            header = f.readline().replace('\n', '')
            cmb = np.loadtxt(f)
        # first column header has the units of the type of opearing points
        cols = misc.remove_items(header.split(']')[1].split(' '), '')
        nrmodes = np.array([int(k) for k in cols]).max()
        # when there is only data for one operating condition we only have one
        # row and consequently only a 1D array
        if len(cmb.shape) == 1:
            cmb = cmb.reshape( (1, cmb.shape[0]) )
        wind = cmb[:,0]
        freq = cmb[:,1:nrmodes+1]
        damp = cmb[:,nrmodes+1:nrmodes*2+1]
        real_eig = None
        if cmb.shape[1] > nrmodes*2+1:
            real_eig = cmb[:,nrmodes*2+1:]

        return wind, freq, damp, real_eig

    def load_cmb_df(self, fname):
        # index name can be rotor speed or wind speed
        with open(fname) as f:
            header = f.readline()
        oper_name = header.split('1')[0].strip().replace('#', '').lower()
        oper_name = oper_name.replace(' ', '').replace('[', '_')[:-1]
        oper_name = oper_name.replace('/', '')

        speed, freq, damp, real_eig = self.load_cmb(fname)
        mods = freq.shape[1]
        ops = freq.shape[0]

        df = pd.DataFrame(columns=[oper_name, 'Fd_hz', 'damp_ratio', 'mode'])
        df['Fd_hz'] = freq.flatten()
        df['damp_ratio'] = damp.flatten()
        if real_eig is not None:
            df['real_eig'] = real_eig.flatten()
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

    def load_ind(self, fname, colwidth=None):
        """for results withouth gradients, colwidth=14, otherwise 28. Set to
        None to derive automatically.
        """
        ind = InductionResults(colwidth=colwidth)
        ind.read(fname)
        return ind.df_data

    def load_amp(self, fname):

        with open(fname) as f:
            line = f.readline()
            f.readline()
            line3 = f.readline()
            line4 = f.readline() # first data entry

        # assuming first we have: "# Mode number:"
        mode_nrs = line.split()[3:]

        elements_line1 = line4.split()
#        nrcols = len(elements_line1)
        element1 = elements_line1[0]
        width_col1 = line4.find(element1) + len(element1)

#        width = 14
#        nrcols = int((len(line)-1)/width)
#        # first columns has one extra character
#        # col nr1: rotor speed, col nr2: radius
#        widths = [width+1] + [width]*(nrcols-1)

        # the rotor/wind speed column does not have a unit
        ilocs = [0, width_col1-1] + [m.start() for m in re.finditer('\]', line3)]
        widths = np.diff(np.array(ilocs))
        # the length of the first element should be +1 due to zero based index
        widths[0] += 1

        # last line is empty
        df = pd.read_fwf(fname, header=2, widths=widths, skipfooter=1)
        units = regex_units.findall(''.join(df.columns))
        # no column number in the column name
        # since U_x, u_y, phase and theta will be repeated as many times as
        # there are modes, add the mode number in the column name
        columns = [k.replace('#', '').strip() for k in df.columns]
#        nrmodes = int((len(columns) - 2 )/6)
#        for k in range(nrmodes):
#            for i in range(6):
#                j = 2+k*6+i
#                columns[j] = columns[j].split('.')[0] + ' nr%i' % (k+1)

        for i, k in enumerate(mode_nrs):
            columns[i+2] = columns[i+2].split('.')[0] + ' nr%s' % k

        df.columns = columns

        return df, units

    def load_operation(self, fname):

        operation = np.loadtxt(fname, skiprows=1)
        # when the array is empty, set operation to an empty DataFrame
        if len(operation) == 0:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm']
            return pd.DataFrame(columns=cols)
        # when there is only one data point, the array is 1D, we allways need
        # a 2D array otherwise the columns become rows in the DataFrame
        elif len(operation.shape) == 1:
            operation = operation.reshape((1, operation.shape[0]))
        try:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm']
            operation = pd.DataFrame(operation, columns=cols)
        except ValueError:
            cols = ['windspeed', 'pitch_deg', 'rotorspeed_rpm', 'P_aero',
                    'T_aero']
            operation = pd.DataFrame(operation, columns=cols)
        return operation

    def load_matrices(self, fpath, basename, operating_point=1,
                      control_mat=False, local_wind_mat=False):
        """Load HAWCStab2 State Space system matrices

        The general file name format is:
        BASENAMETYPE_ase_ops_OPERATING_POINT_NUMBER.dat

        Where TYPE can be of the following:
            * amat, bmat, bvmat, cmat, dmat, dvmat, emat, fmat, fvmat

        Additionally, when including the control matrices:
            * BASENAMETYPE_ops_OPERATING_POINT_NUMBER.dat
            * TYPE: acmat, bcmat, ccmat, dcmat

        Or when including local wind speed
            * BASENAMETYPE_ase_ops_OPERATING_POINT_NUMBER.dat
            * TYPE: bvmat_loc_v, dvmat_loc_v, fvmat_loc_v


        Parameters
        ----------

        fpath : str

        basename : str

        operating_point : int, default=1


        Returns
        -------

        matrices : dict

        """
        mnames = ['amat', 'bmat', 'bvmat', 'cmat', 'dmat', 'dvmat', 'emat',
                  'fmat', 'fvmat']
        mnames_c = ['acmat', 'bcmat', 'ccmat', 'dcmat']
        mnames_v = ['bvmat_loc_v', 'dvmat_loc_v', 'fvmat_loc_v']

        if control_mat:
            mnames += mnames_c
        if local_wind_mat:
            mnames += mnames_v

        matrices = {}

        ase_template = '{:s}{:s}_ase_ops_{:d}.dat'
        ops_template = '{:s}{:s}_ops_{:d}.dat'

        for mname in mnames:
            rpl = (basename, mname, operating_point)
            template = ase_template
            if mname in mnames_c:
                template = ops_template
            fname = os.path.join(fpath, template.format(*rpl))
            matrices[mname] = np.loadtxt(fname)

        return matrices

    def write_ae_sections_h2(self):
        """
        Get the aerosection positions from the HS2 ind result file and
        write them as outputs for HAWC2

        NOT IMPLEMENTED YET
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
            ax.plot(res.wind, res.power, color=colors[i],
                    label='Power %s ' % labels[i],
                    marker=symbols[i], ls='-', alpha=alphas[i])
            ax.set_title('Aerodynamic Power [kW]')#, RPM')

            ax = axes[0,1]
            ax.plot(res.wind, res.pitch_deg, color=colors[i],
                    label='Pitch %s' % labels[i],
                    marker=symbols[i], ls='-', alpha=alphas[i])
            ax.plot(res.wind, res.rpm, color=colors[i],
                    label='RPM %s ' % labels[i],
                    marker=symbols[i], ls='--', alpha=alphas[i])
            ax.set_title('Pitch [deg], RPM')

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


class ReadControlTuning(object):

    def __init__(self):
        """
        """
        self._aerogains = False

    def parse_line(self, line, controller):
        """Parses the output lines with the controller tuning parameters.
        Does not parse the aerodynamic gain lines.
        """

        if line.startswith('Aerodynamic gains'):
            self._aerogains = True
            return

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
                elif not self._aerogains:
                    self.parse_line(line, controller)
                # do not break since we want to see if aero gains are included
                # elif self._aerogains:
                #     break

        self.aero_gains_units = ['[deg]', '[kNm/deg]', '[kNm/deg]',
                                 '[kNm/(rad/s)]', '[kNm/(rad/s)]']
        self.aero_gains = pd.DataFrame()
        # in case the gains are missing from the file, don't try to read it
        if i > 17:
            arr = np.loadtxt(fpath, skiprows=17)
            columns = ['theta', 'dq/dtheta', 'dq/dtheta_fit', 'dq/domega',
                       'dq/domega_fit']
            self.aero_gains = pd.DataFrame(arr, columns=columns)

        # set some parameters to zero for the linear case, or when aerodynamic
        # gain scheduling is not used
        if not hasattr(self.pi_gen_reg2, 'Kd'):
            setattr(self.pi_gen_reg2, 'Kd', 0.0)
        if not hasattr(self.pi_pitch_reg3, 'Kd'):
            setattr(self.pi_pitch_reg3, 'Kd', 0.0)
        if not hasattr(self.pi_pitch_reg3, 'K2'):
            setattr(self.pi_pitch_reg3, 'K2', 0.0)
        if not hasattr(self.aero_damp, 'Kp2'):
            setattr(self.aero_damp, 'Kp2', 0.0)
        if not hasattr(self.aero_damp, 'Ko1'):
            setattr(self.aero_damp, 'Ko1', 0.0)
        if not hasattr(self.aero_damp, 'Ko2'):
            setattr(self.aero_damp, 'Ko2', 0.0)

    def parameters2tags(self):
        """Convert the tuning parameters into a dictionary whos keys are
        compatible with tag names in a HAWC2 master file.
        """

        tune_tags = {}

        tune_tags['[pi_gen_reg1.K]'] = self.pi_gen_reg1.K

        tune_tags['[pi_gen_reg2.I]'] = self.pi_gen_reg2.I
        tune_tags['[pi_gen_reg2.Kp]'] = self.pi_gen_reg2.Kp
        tune_tags['[pi_gen_reg2.Ki]'] = self.pi_gen_reg2.Ki
        tune_tags['[pi_gen_reg2.Kd]'] = self.pi_gen_reg2.Kd

        tune_tags['[pi_pitch_reg3.Kp]'] = self.pi_pitch_reg3.Kp
        tune_tags['[pi_pitch_reg3.Ki]'] = self.pi_pitch_reg3.Ki
        tune_tags['[pi_pitch_reg3.Kd]'] = self.pi_pitch_reg3.Kd
        tune_tags['[pi_pitch_reg3.K1]'] = self.pi_pitch_reg3.K1
        tune_tags['[pi_pitch_reg3.K2]'] = self.pi_pitch_reg3.K2

        tune_tags['[aero_damp.Kp2]'] = self.aero_damp.Kp2
        tune_tags['[aero_damp.Ko1]'] = self.aero_damp.Ko1
        tune_tags['[aero_damp.Ko2]'] = self.aero_damp.Ko2

        return tune_tags


if __name__ == '__main__':

    pass
