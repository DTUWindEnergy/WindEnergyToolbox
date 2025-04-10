# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:53:59 2014

@author: dave
"""
__author__ = 'David Verelst'
__license__ = 'GPL'
__version__ = '0.5'

import os
import math
from time import time
from itertools import chain

import numpy as np
import scipy as sp
import scipy.io as sio
# scipy changed interface name from 1.14 to 1.15
try:
    from scipy.integrate import trapezoid # scipy >1.15
except ImportError:
    from scipy.integrate import trapz as trapezoid  # scipy <=1.14
import pandas as pd

# misc is part of prepost, which is available on the dtu wind gitlab server:
# https://gitlab.windenergy.dtu.dk/dave/prepost
from wetb.prepost import misc
# wind energy python toolbox, available on the dtu wind redmine server:
# http://vind-redmine.win.dtu.dk/projects/pythontoolbox/repository/show/fatigue_tools
from wetb.hawc2.sensor_names import unified_channel_names
from wetb.hawc2.Hawc2io import ReadHawc2
from wetb.fatigue_tools.fatigue import (eq_load, cycle_matrix2)


class LogFile(object):
    """Check a HAWC2 log file for errors.
    """

    def __init__(self):

        # the total message list log:
        self.MsgListLog = []
        # a smaller version, just indication if there are errors:
        self.MsgListLog2 = dict()

        # specify which message to look for. The number track's the order.
        # this makes it easier to view afterwards in spreadsheet:
        # every error will have its own column

        # error messages that appear during initialisation
        self.err_init = {}
        self.err_init[' *** ERROR *** Error in com'] = len(self.err_init)
        self.err_init[' *** ERROR ***  in command '] = len(self.err_init)
        #  *** WARNING *** A comma "," is written within the command line
        self.err_init[' *** WARNING *** A comma ",'] = len(self.err_init)
        #  *** ERROR *** Not correct number of parameters
        self.err_init[' *** ERROR *** Not correct '] = len(self.err_init)
        #  *** INFO *** End of file reached
        self.err_init[' *** INFO *** End of file r'] = len(self.err_init)
        #  *** ERROR *** No line termination in command line
        self.err_init[' *** ERROR *** No line term'] = len(self.err_init)
        #  *** ERROR *** MATRIX IS NOT DEFINITE
        self.err_init[' *** ERROR *** MATRIX IS NO'] = len(self.err_init)
        #  *** ERROR *** There are unused relative
        self.err_init[' *** ERROR *** There are un'] = len(self.err_init)
        #  *** ERROR *** Error finding body based
        self.err_init[' *** ERROR *** Error findin'] = len(self.err_init)
        #  *** ERROR *** In body actions
        self.err_init[' *** ERROR *** In body acti'] = len(self.err_init)
        #  *** ERROR *** Command unknown and ignored
        self.err_init[' *** ERROR *** Command unkn'] = len(self.err_init)
        #  *** ERROR *** ERROR - More bodies than elements on main_body: tower
        self.err_init[' *** ERROR *** ERROR - More'] = len(self.err_init)
        #  *** ERROR *** The program will stop
        self.err_init[' *** ERROR *** The program '] = len(self.err_init)
        #  *** ERROR *** Unknown begin command in topologi.
        self.err_init[' *** ERROR *** Unknown begi'] = len(self.err_init)
        #  *** ERROR *** Not all needed topologi main body commands present
        self.err_init[' *** ERROR *** Not all need'] = len(self.err_init)
        #  *** ERROR ***  opening timoschenko data file
        self.err_init[' *** ERROR ***  opening tim'] = len(self.err_init)
        #  *** ERROR *** Error opening AE data file
        self.err_init[' *** ERROR *** Error openin'] = len(self.err_init)
        #  *** ERROR *** Requested blade _ae set number not found in _ae file
        self.err_init[' *** ERROR *** Requested bl'] = len(self.err_init)
        #  Error opening PC data file
        self.err_init[' Error opening PC data file'] = len(self.err_init)
        #  *** ERROR *** error reading mann turbulence
        self.err_init[' *** ERROR *** error readin'] = len(self.err_init)
#        #  *** INFO *** The DLL subroutine
#        self.err_init[' *** INFO *** The DLL subro'] = len(self.err_init)
        #  ** WARNING: FROM ESYS ELASTICBAR: No keyword
        self.err_init[' ** WARNING: FROM ESYS ELAS'] = len(self.err_init)
        #  *** ERROR *** DLL ./control/killtrans.dll could not be loaded - error!
        self.err_init[' *** ERROR *** DLL'] = len(self.err_init)
        # *** ERROR *** The DLL subroutine
        self.err_init[' *** ERROR *** The DLL subr'] = len(self.err_init)
        # *** ERROR *** Mann turbulence length scale must be larger than zero!
        # *** ERROR *** Mann turbulence alpha eps value must be larger than zero!
        # *** ERROR *** Mann turbulence gamma value must be larger than zero!
        self.err_init[' *** ERROR *** Mann turbule'] = len(self.err_init)

        # *** WARNING *** Shear center x location not in elastic center, set to zero
        self.err_init[' *** WARNING *** Shear cent'] = len(self.err_init)
        # Turbulence file ./xyz.bin does not exist
        self.err_init[' Turbulence file '] = len(self.err_init)
        self.err_init[' *** WARNING ***'] = len(self.err_init)
        self.err_init[' *** ERROR ***'] = len(self.err_init)
        self.err_init[' WARNING'] = len(self.err_init)
        self.err_init[' ERROR'] = len(self.err_init)

        # error messages that appear during simulation
        self.err_sim = {}
        #  *** ERROR *** Wind speed requested inside
        self.err_sim[' *** ERROR *** Wind speed r'] = len(self.err_sim)
        #  Maximum iterations exceeded at time step:
        self.err_sim[' Maximum iterations exceede'] = len(self.err_sim)
        #  Solver seems not to converge:
        self.err_sim[' Solver seems not to conver'] = len(self.err_sim)
        #  *** ERROR *** Out of x bounds:
        self.err_sim[' *** ERROR *** Out of x bou'] = len(self.err_sim)
        #  *** ERROR *** Out of limits in user defined shear field - limit value used
        self.err_sim[' *** ERROR *** Out of limit'] = len(self.err_sim)

        # NEAR WAKE ERRORS
        # ERROR in Near Wake! The radius of the tip is smaller (or equal) to
        self.err_sim[' ERROR in Near Wake! The ra'] = len(self.err_sim)
        # ERROR: Maximum number of near wake iterations reached
        self.err_sim[' ERROR: Maximum number of n'] = len(self.err_sim)

        # TODO: error message from a non existing channel output/input
        # add more messages if required...

        self.init_cols = len(self.err_init)
        self.sim_cols = len(self.err_sim)
        self.header = None

    def readlog(self, fname, case=None, save_iter=False):
        """
        """
        # be cautious and try a few encodings when reading the file
        lines = misc.readlines_try_encodings(fname)

        # keep track of the messages allready found in this file
        tempLog = []
        tempLog.append(fname)
        exit_correct, found_error = False, False

        subcols_sim = 4
        subcols_init = 2
        # create empty list item for the different messages and line
        # number. Include one column for non identified messages
        for j in range(self.init_cols):
            # 2 sub-columns per message: nr, msg
            for k in range(subcols_init):
                tempLog.append('')
        for j in range(self.sim_cols):
            # 4 sub-columns per message: first, last, nr, msg
            for k in range(subcols_sim):
                tempLog.append('')
        # and two more columns at the end for messages of unknown origin
        tempLog.append('')
        tempLog.append('')

        # if there is a cases object, see how many time steps we expect
        if case is not None:
            dt = float(case['[dt_sim]'])
            time_steps = int(float(case['[time_stop]']) / dt)
            iterations = np.ndarray( (time_steps+1,3), dtype=np.float32 )
        else:
            iterations = np.ndarray( (len(lines),3), dtype=np.float32 )
            dt = False
        iterations[:,0:2] = np.nan
        iterations[:,2] = 0

        # keep track of the time_step number
        time_step, init_block = 0, True
        # check for messages in the current line
        # for speed: delete from message watch list if message is found
        for j, line in enumerate(lines):
            # all id's of errors are 27 characters long
            msg = line[:27]
            # remove the line terminator, this seems to take 2 characters
            # on PY2, but only one in PY3
            line = line.replace('\n', '')

            # keep track of the number of iterations
            if line[:12] == ' Global time':
                iterations[time_step,0] = float(line[14:40])
                # for PY2, new line is 2 characters, for PY3 it is one char
                iterations[time_step,1] = int(line[-6:])
                # time step is the first time stamp
                if not dt:
                    dt = float(line[15:40])
                time_step += 1
                # no need to look for messages if global time is mentioned
                continue

            elif line[:4] == ' kfw':
                pass
            # Global time =    17.7800000000000      Iter =            2
            # kfw  0.861664060457402
            #  nearwake iterations          17

            # computed relaxation factor  0.300000000000000


            elif line[:20] == ' Starting simulation':
                init_block = False

            elif init_block:
                # if string is shorter, we just get a shorter string.
                # checking presence in dict is faster compared to checking
                # the length of the string
                # first, last, nr, msg
                if msg in self.err_init:
                    # icol=0 -> fname
                    icol = subcols_init*self.err_init[msg] + 1
                    # 0: number of occurances
                    if tempLog[icol] == '':
                        tempLog[icol] = '1'
                    else:
                        tempLog[icol] = str(int(tempLog[icol]) + 1)
                    # 1: the error message itself
                    tempLog[icol+1] = line
                    found_error = True

            # find errors that can occur during simulation
            elif msg in self.err_sim:
                icol = subcols_sim*self.err_sim[msg]
                icol += subcols_init*self.init_cols + 1
                # 1: time step of first occurance
                if tempLog[icol]  == '':
                    tempLog[icol] = '%i' % time_step
                # 2: time step of last occurance
                tempLog[icol+1] = '%i' % time_step
                # 3: number of occurances
                if tempLog[icol+2] == '':
                    tempLog[icol+2] = '1'
                else:
                    tempLog[icol+2] = str(int(tempLog[icol+2]) + 1)
                # 4: the error message itself
                tempLog[icol+3] = line

                found_error = True
                iterations[time_step-1,2] = 1

            # method of last resort, we have no idea what message
            elif line[:10] == ' *** ERROR' or line[:10]==' ** WARNING' \
                    or line[:6] == ' ERROR':
                icol = subcols_sim*self.sim_cols
                icol += subcols_init*self.init_cols + 1
                # line number of the message
                tempLog[icol] = j
                # and message
                tempLog[icol+1] = line
                found_error = True
                iterations[time_step-1,2] = 1

        # remove not-used rows from iterations
        iterations = iterations[:time_step,:]

        # simulation and simulation output time based on the tags
        # FIXME: ugly, do not mix tags with what is actually happening in the
        # log files!!
        if case is not None:
            t_stop = float(case['[time_stop]'])
            duration = float(case['[duration]'])
        else:
            t_stop = np.nan
            duration = -1

        # if no time steps have passed
        if iterations.shape == (0,3):
            elapsed_time = -1
            tempLog.append('')
        # see if the last line holds the sim time
        elif line[:15] ==  ' Elapsed time :':
            exit_correct = True
            elapsed_time = float(line[15:-1])
            tempLog.append( elapsed_time )
        # in some cases, Elapsed time is not given, and the last message
        # might be: " Closing of external type2 DLL"
        elif line[:20] == ' Closing of external':
            exit_correct = True
            elapsed_time = iterations[time_step-1,0]
            tempLog.append( elapsed_time )
        # FIXME: this is weird mixing of referring to t_stop from the tags
        # and the actual last recorded time step
        elif np.allclose(iterations[time_step-1,0], t_stop):
            exit_correct = True
            elapsed_time = iterations[time_step-1,0]
            tempLog.append( elapsed_time )
        else:
            elapsed_time = -1
            tempLog.append('')

        if iterations.shape == (0,3):
            last_time_step = 0
        else:
            last_time_step = iterations[time_step-1,0]

        # give the last recorded time step
        tempLog.append('%1.11f' % last_time_step)
        # simulation_time, as taken from cases
        tempLog.append('%1.01f' % t_stop)
        # real_sim_time
        tempLog.append('%1.04f' % (last_time_step/elapsed_time))
        tempLog.append('%1.01f' % duration)

        # as last element, add the total number of iterations
        itertotal = np.nansum(iterations[:,1])
        tempLog.append('%1.0f' % itertotal)

        # the delta t used for the simulation
        if dt:
            tempLog.append('%1.7f' % dt)
        else:
            tempLog.append('nan')

        # number of time steps
        tempLog.append('%i' % (time_step))

        # if the simulation didn't end correctly, the elapsed_time doesn't
        # exist. Add the average and maximum nr of iterations per step
        # or, if only the structural and eigen analysis is done, we have 0
        try:
            ratio = float(elapsed_time)/float(itertotal)
            # FIXME: this needs to be fixed proper while testing the analysis
            # of various log files and edge cases
            if elapsed_time < 0:
                tempLog.append('')
            else:
                tempLog.append('%1.6f' % ratio)
        except (UnboundLocalError, ZeroDivisionError, ValueError) as e:
            tempLog.append('')
        # when there are no time steps (structural analysis only)
        try:
            tempLog.append('%1.2f' % iterations[:,1].mean())
            tempLog.append('%1.2f' % iterations[:,1].max())
        except ValueError:
            tempLog.append('')
            tempLog.append('')

        # FIXME: we the sim crashes at generating the turbulence box
        # there is one element too much at the end
        tempLog = tempLog[:len(self._header().split(';'))]

        # save the iterations in the results folder
        if save_iter:
            fiter = os.path.basename(fname).replace('.log', '.iter')
            fmt = ['%12.06f', '%4i', '%4i']
            if case is not None:
                fpath = os.path.join(case['[run_dir]'], case['[iter_dir]'])
                # in case it has subdirectories
                for tt in [3,2,1]:
                    tmp = os.path.sep.join(fpath.split(os.path.sep)[:-tt])
                    if not os.path.exists(tmp):
                        os.makedirs(tmp)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                np.savetxt(fpath + fiter, iterations, fmt=fmt)
            else:
                logpath = os.path.dirname(fname)
                np.savetxt(os.path.join(logpath, fiter), iterations, fmt=fmt)

        # append the messages found in the current file to the overview log
        self.MsgListLog.append(tempLog)
        self.MsgListLog2[fname] = [found_error, exit_correct]

    def _msglistlog2csv(self, contents):
        """Write LogFile.MsgListLog to a csv file. Use LogFile._header to
        create a header.
        """
        for k in self.MsgListLog:
            for n in k:
                contents = contents + str(n) + ';'
            # at the end of each line, new line symbol
            contents = contents + '\n'
        return contents

    def csv2df(self, fname, header=0):
        """Read a csv log file analysis and convert to a pandas.DataFrame
        """
        colnames, min_itemsize, dtypes = self.headers4df()
        df = pd.read_csv(fname, header=header, names=colnames, sep=';')
        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)
            # replace nan with empty for str columns
            if dtype == str:
                df[col] = df[col].str.replace('nan', '')
        return df

    def _header(self):
        """Header for log analysis csv file
        """

        # write the results in a file, start with a header
        contents = 'file name;' + 'nr;msg;'*(self.init_cols)
        contents += 'first_tstep;last_tstep;nr;msg;'*(self.sim_cols)
        contents += 'lnr;msg;'
        # and add headers for elapsed time, nr of iterations, and sec/iteration
        contents += 'Elapsted time;last time step;Simulation time;'
        contents += 'real sim time;Sim output time;'
        contents += 'total iterations;dt;nr time steps;'
        contents += 'seconds/iteration;average iterations/time step;'
        contents += 'maximum iterations/time step;\n'

        return contents

    def headers4df(self):
        """Create header and a minimum itemsize for string columns when
        converting a Log check analysis to a pandas.DataFrame

        Returns
        -------

        header : list
            List of column names as generated by WindIO.LogFile._header

        min_itemsize : dict
            Dictionary with column names as keys, and the minimum string lenght
            as values.

        dtypes : dict
            Dictionary with column names as keys, and data types as values
        """
        chain_iter = chain.from_iterable

        nr_init = len(self.err_init)
        nr_sim = len(self.err_sim)

        colnames = ['file_name']
        colnames.extend(list(chain_iter(('nr_%i' % i, 'msg_%i' % i)
                      for i in range(nr_init))) )

        gr = ('first_tstep_%i', 'last_step_%i', 'nr_%i', 'msg_%i')
        colnames.extend(list(chain_iter( (k % i for k in gr)
                           for i in range(100,100+nr_sim,1))) )
        colnames.extend(['nr_extra', 'msg_extra'])
        colnames.extend(['elapsted_time',
                       'last_time_step',
                       'simulation_time',
                       'real_sim_time',
                       'sim_output_time',
                       'total_iterations',
                       'dt',
                       'nr_time_steps',
                       'seconds_p_iteration',
                       'mean_iters_p_time_step',
                       'max_iters_p_time_step',
                       'sim_id'])
        dtypes = {}

        # str and float datatypes for
        msg_cols = ['msg_%i' % i for i in range(nr_init-1)]
        msg_cols.extend(['msg_%i' % i for i in range(100,100+nr_sim,1)])
        msg_cols.append('msg_extra')
        dtypes.update({k:str for k in msg_cols})
        # make the message/str columns long enough
        min_itemsize = {'msg_%i' % i : 100 for i in range(nr_init-1)}

        # column names holding the number of occurances of messages
        nr_cols = ['nr_%i' % i for i in range(nr_init-1)]
        nr_cols.extend(['nr_%i' % i for i in range(100,100+nr_sim,1)])
        # other float values
        nr_cols.extend(['elapsted_time', 'total_iterations'])
        # NaN only exists in float arrays, not integers (NumPy limitation)
        # so use float instead of int
        dtypes.update({k:np.float64 for k in nr_cols})

        return colnames, min_itemsize, dtypes


class LoadResults(ReadHawc2):
    """Read a HAWC2 result data file

    Usage:
    obj = LoadResults(file_path, file_name)

    This is a subclass of wetb.hawc2.Windio:ReadHawc2.

    Available output:
    obj.sig[timeStep,channel]   : complete result file in a numpy array
    obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
    obj.error_msg: is 'none' if everything went OK, otherwise it holds the
    error

    The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos'] = pos
                ch_dict[tag]['sensortype']
                ch_dict[tag]['component']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the DLL's this is:
            key:
                DLL-dll_name-io-io_nr
                DLL-yaw_control-outvec-3
                DLL-yaw_control-inpvec-1
            value:
                ch_dict[tag]['dll_name']
                ch_dict[tag]['io']
                ch_dict[tag]['io_nr']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the bearings this is:
            key:
                bearing-bearing_name-output_type-units
                bearing-shaft_nacelle-angle_speed-rpm
            value:
                ch_dict[tag]['bearing_name']
                ch_dict[tag]['output_type']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']

    """

    # start with reading the .sel file, containing the info regarding
    # how to read the binary file and the channel information
    def __init__(self, file_path, file_name, debug=False, usecols=None,
                 readdata=True):

        self.debug = debug

        # timer in debug mode
        if self.debug:
            start = time()

        self.file_path = file_path
        # remove .log, .dat, .sel extensions who might be accedental left
        ext = file_name.split('.')[-1]
        if ext in ['htc', 'sel', 'dat', 'log', 'hdf5']:
            file_name = file_name.replace('.' + ext, '')
        # FIXME: since HAWC2 will always have lower case output files, convert
        # any wrongly used upper case letters to lower case here
        self.file_name = file_name
        FileName = os.path.join(self.file_path, self.file_name)

        super(LoadResults, self).__init__(FileName, ReadOnly=readdata)
        self.FileType = self.FileFormat
        if self.FileType.find('HAWC2_') > -1:
            self.FileType = self.FileType[6:]

        if readdata:
            ChVec = [] if usecols is None else usecols
            self.sig = self.ReadAll(ChVec=ChVec)

        # info in sel file is not available when not reading gtsdf
        # so this is only skipped when readdata is false and FileType is gtsdf
        if not (not readdata and (self.FileType == 'GTSDF')):
            self.N = int(self.NrSc)
            self.Nch = int(self.NrCh)
            self.ch_details = np.ndarray(shape=(self.Nch, 3), dtype='<U150')
            for ic in range(self.Nch):
                self.ch_details[ic, 0] = self.ChInfo[0][ic]
                self.ch_details[ic, 1] = self.ChInfo[1][ic]
                self.ch_details[ic, 2] = self.ChInfo[2][ic]

        self._unified_channel_names()

        if self.debug:
            stop = time() - start
            print('time to load HAWC2 file:', stop, 's')

    def _unified_channel_names(self):
        """For backwards compatibiliity: to create an alternative sensor naming
        scheme that is consistent and unique so you can always refer to a channel
        name instead of its index in an output file.

        See wetb.hawc2.sensor_names.unified_channel_names instead.

        Returns
        -------
        None.

        """

        self.ch_dict, self.ch_df = unified_channel_names(self.ChInfo)

    def _ch_dict2df(self):
        """
        Create a DataFrame version of the ch_dict, and the chi columns is
        set as the index
        """
        # identify all the different columns
        cols = set()
        for ch_name, channelinfo in self.ch_dict.items():
            cols.update(set(channelinfo.keys()))

        df_dict = {col: [] for col in cols}
        df_dict['unique_ch_name'] = []
        for ch_name, channelinfo in self.ch_dict.items():
            cols_ch = set(channelinfo.keys())
            for col in cols_ch:
                df_dict[col].append(channelinfo[col])
            # the remainder columns we have not had yet. Fill in blank
            for col in (cols - cols_ch):
                df_dict[col].append('')
            df_dict['unique_ch_name'].append(ch_name)

        self.ch_df = pd.DataFrame(df_dict)
        self.ch_df.set_index('chi', inplace=True)

    def _data_window(self, nr_rev=None, time=None):
        """
        Based on a time interval, create a proper slice object
        ======================================================

        The window will start at zero and ends with the covered time range
        of the time input.

        Paramters
        ---------

        nr_rev : int, default=None
            NOT IMPLEMENTED YET

        time : list, default=None
            time = [time start, time stop]

        Returns
        -------

        slice_

        window

        zoomtype

        time_range
            time_range = [0, time[1]]

        """

        # -------------------------------------------------
        # determine zome range if necesary
        # -------------------------------------------------
        time_range = None
        if nr_rev:
            raise NotImplementedError
            # input is a number of revolutions, get RPM and sample rate to
            # calculate the required range
            # TODO: automatich detection of RPM channel!
            time_range = nr_rev/(self.rpm_mean/60.)
            # convert to indices instead of seconds
            i_range = int(self.Freq*time_range)
            window = [0, time_range]
            # in case the first datapoint is not at 0 seconds
            i_zero = int(self.sig[0, 0]*self.Freq)
            slice_ = np.r_[i_zero:i_range+i_zero]

            zoomtype = '_nrrev_' + format(nr_rev, '1.0f') + 'rev'

        elif time.any():
            time_range = time[1] - time[0]

            i_start = int(time[0]*self.Freq)
            i_end = int(time[1]*self.Freq)
            slice_ = np.r_[i_start:i_end]
            window = [time[0], time[1]]

            zoomtype = '_zoom_%1.1f-%1.1fsec' % (time[0], time[1])

        return slice_, window, zoomtype, time_range

    def sig2df(self):
        """Convert sig to dataframe with unique channel names as column names.
        """
        # channels that are not part of the naming scheme are not included
        df = pd.DataFrame(self.sig[:,self.ch_df.index],
                          columns=self.ch_df['unique_ch_name'])

        return df

    # TODO: general signal method, this is not HAWC2 specific, move out
    def calc_stats(self, sig, i0=0, i1=None):

        stats = {}
        # calculate the statistics values:
        stats['max'] = sig[i0:i1, :].max(axis=0)
        stats['min'] = sig[i0:i1, :].min(axis=0)
        stats['mean'] = sig[i0:i1, :].mean(axis=0)
        stats['std'] = sig[i0:i1, :].std(axis=0)
        stats['range'] = stats['max'] - stats['min']
        stats['absmax'] = np.absolute(sig[i0:i1, :]).max(axis=0)
        stats['rms'] = np.sqrt(np.mean(sig[i0:i1, :]*sig[i0:i1, :], axis=0))
        stats['int'] = trapezoid(sig[i0:i1, :], x=sig[i0:i1, 0], axis=0)
        return stats

    def statsdel_df(self, i0=0, i1=None, statchans='all', delchans='all',
                    m=[3, 4, 6, 8, 10, 12], neq=None, no_bins=46):
        """Calculate statistics and equivalent loads for the current loaded
        signal.

        Parameters
        ----------

        i0 : int, default=0

        i1 : int, default=None

        channels : list, default='all'
            all channels are selected if set to 'all', otherwise define a list
            using the unique channel defintions.

        neq : int, default=1

        no_bins : int, default=46

        Return
        ------

        statsdel : pd.DataFrame
            Pandas DataFrame with the statistical parameters and the different
            fatigue coefficients as columns, and channels as rows. As index the
            unique channel name is used.

        """

        stats = ['max', 'min', 'mean', 'std', 'range', 'absmax', 'rms', 'int']
        if statchans == 'all':
            statchans = self.ch_df['unique_ch_name'].tolist()
            statchis = self.ch_df['unique_ch_name'].index.values
        else:
            sel = self.ch_df['unique_ch_name']
            statchis = self.ch_df[sel.isin(statchans)].index.values

        if delchans == 'all':
            delchans = self.ch_df['unique_ch_name'].tolist()
            delchis = self.ch_df.index.values
        else:
            sel = self.ch_df['unique_ch_name']
            delchis = self.ch_df[sel.isin(delchans)].index.values

        # delchans has to be a subset of statchans!
        if len(set(delchans) - set(statchans)) > 0:
            raise ValueError('delchans has to be a subset of statchans')

        tmp = np.ndarray((len(statchans), len(stats+m)))
        tmp[:,:] = np.nan
        m_cols = ['m=%i' % m_ for m_ in m]
        statsdel = pd.DataFrame(tmp, columns=stats+m_cols)
        statsdel.index = statchans

        datasel = self.sig[i0:i1,statchis]
        time = self.sig[i0:i1,0]
        statsdel['max'] = datasel.max(axis=0)
        statsdel['min'] = datasel.min(axis=0)
        statsdel['mean'] = datasel.mean(axis=0)
        statsdel['std'] = datasel.std(axis=0)
        statsdel['range'] = statsdel['max'] - statsdel['min']
        statsdel['absmax'] = np.abs(datasel).max(axis=0)
        statsdel['rms'] = np.sqrt(np.mean(datasel*datasel, axis=0))
        statsdel['int'] = trapezoid(datasel, x=time, axis=0)
        statsdel['intabs'] = trapezoid(np.abs(datasel), x=time, axis=0)

        if neq is None:
            neq = self.sig[-1,0] - self.sig[0,0]

        for chi, chan in zip(delchis, delchans):
            signal = self.sig[i0:i1,chi]
            eq = self.calc_fatigue(signal, no_bins=no_bins, neq=neq, m=m)
            statsdel.loc[chan, m_cols] = eq

        return statsdel

    # TODO: general signal method, this is not HAWC2 specific, move out
    def calc_fatigue(self, signal, no_bins=46, m=[3, 4, 6, 8, 10, 12], neq=1):
        """
        Parameters
        ----------

        signal: 1D array
            One dimentional array containing the signal.
        no_bins: int
            Number of bins for the binning of the amplitudes.
        m: list
            Values of the slope of the SN curve.
        neq: int
            Number of equivalent cycles

        Returns
        -------
        eq: list
            Damage equivalent loads for each m value.
        """

        return eq_load(signal, no_bins=no_bins, m=m, neq=neq)[0]

    def cycle_matrix(self, signal, no_bins=46):
        """Cycle/Markov matrix.

        Convenience function for wetb.fatigue_tools.fatigue.cycle_matrix2

        Parameters
        ----------

        signal: 1D array
            One dimentional array containing the signal.

        no_bins: int
            Number of bins for the binning of the amplitudes.

        Returns
        -------

        cycles : ndarray, shape(ampl_bins, mean_bins)
            A bi-dimensional histogram of load cycles(full cycles). Amplitudes
            are histogrammed along the first dimension and mean values are
            histogrammed along the second dimension.

        ampl_edges : ndarray, shape(no_bins+1,n)
            The amplitude bin edges

        mean_edges : ndarray, shape(no_bins+1,n)
            The mean bin edges

        """
        return cycle_matrix2(signal, no_bins)

    def blade_deflection(self):
        """
        """

        # select all the y deflection channels
        db = misc.DictDB(self.ch_dict)

        db.search({'sensortype': 'state pos', 'component': 'z'})
        # sort the keys and save the mean values to an array/list
        chiz, zvals = [], []
        for key in sorted(db.dict_sel.keys()):
            zvals.append(-self.sig[:, db.dict_sel[key]['chi']].mean())
            chiz.append(db.dict_sel[key]['chi'])

        db.search({'sensortype': 'state pos', 'component': 'y'})
        # sort the keys and save the mean values to an array/list
        chiy, yvals = [], []
        for key in sorted(db.dict_sel.keys()):
            yvals.append(self.sig[:, db.dict_sel[key]['chi']].mean())
            chiy.append(db.dict_sel[key]['chi'])

        return np.array(zvals), np.array(yvals)

    def add_channel(self, data, name, units, description='', options=None):
        """Add a channel to self.sig and self.ch_df such that self.statsdel_df
        also calculates the statistics for this channel.

        Parameters
        ----------

        data : np.ndarray(n, 1)
            Array containing the new channel. Should be of shape (n, 1). If not
            it will be reshaped to (len(data),1).

        name : str
            Unique name of the new channel

        units : str
            Channel units

        description : str, default=''
            channel description
        """
        # valid keys for self.res.ch_df
        # add = {'radius':np.nan, 'bearing_name':'', 'azimuth':np.nan, 'coord':'',
        #       'sensortype':'', 'io_nr':np.nan, 'wake_source_nr':np.nan,
        #       'dll':'', 'direction':'', 'blade_nr':np.nan, 'bodyname':'',
        #       'pos':'', 'flap_nr':'', 'sensortag':'', 'component':'', 'units':'',
        #       'io':'', 'unique_ch_name':'new_channel'}

        add = {k:'' for k in self.ch_df.columns}
        if options is not None:
            add.update(options)
        add['unique_ch_name'] = name
        row = [add[k] for k in self.ch_df.columns]

        # add the meta-data to ch_df and ch_details
        self.ch_df.loc[len(self.ch_df)] = row
        cols = [[name, units, description]]
        self.ch_details = np.append(self.ch_details, cols, axis=0)

        # and add to the results array
        if data.shape != (len(data),1):
            data = data.reshape(len(data),1)
        self.sig = np.append(self.sig, data, axis=1)

    def save_chan_names(self, fname):
        """Save unique channel names to text file.
        """
        channels = self.ch_df.ch_name.values
        channels.sort()
        np.savetxt(fname, channels, fmt='%-100s')

    def save_channel_info(self, fname):
        """Save all channel info: unique naming + HAWC2 description from *.sel.
        """
        p1 = self.ch_df.copy()
        # but ignore the units column, we already have that
        p2 = pd.DataFrame(self.ch_details,
                            columns=['Description1', 'units', 'Description2'])
        # merge on the index
        tmp = pd.merge(p1, p2, right_index=True, how='outer', left_index=True)
        tmp.to_excel(fname)

        # for a fixed-with text format instead of csv
#        header = ''.join(['%100s' % k for k in tmp.columns])
#        header = '  windspeed' + header
#        np.savetxt(fname, tmp.to_records(), header=header,
#                   fmt='% 01.06e  ')

        return tmp

    def load_chan_names(self, fname):
        dtype = np.dtype('U100')
        return np.genfromtxt(fname, dtype=dtype, delimiter=';').tolist()

    def save_csv(self, fname, fmt='%.18e', delimiter=','):
        """
        Save to csv and use the unified channel names as columns
        """
        map_sorting = {}
        # first, sort on channel index
        for ch_key, ch in self.ch_dict.items():
            map_sorting[ch['chi']] = ch_key

        header = []
        # not all channels might be present...iterate again over map_sorting
        for chi in map_sorting:
            try:
                sensortag = self.ch_dict[map_sorting[chi]]['sensortag']
                header.append(map_sorting[chi] + ' // ' + sensortag)
            except:
                header.append(map_sorting[chi])

        # and save
        print('saving...', end='')
        np.savetxt(fname, self.sig[:, list(map_sorting.keys())], fmt=fmt,
                   delimiter=delimiter, header=delimiter.join(header))
        print(fname)

    def save_df(self, fname):
        """
        Save the HAWC2 data and sel file in a DataFrame that contains all the
        data, and all the channel information (the one from the sel file
        and the parsed from this function)
        """

        self.sig
        self.ch_details
        self.ch_dict

    def save_matlab(self, fname):
        """Save output in Matlab format.
        """
        # all channels
        details = np.zeros((self.sig.shape[1],4), dtype=np.object)
        for i in range(self.sig.shape[1]):
            details[i,0:3] = self.ch_details[i,:]
            details[i,3] = self.ch_df.loc[i,'unique_ch_name']
        sio.savemat(fname, {'sig':self.sig, 'description':details})


def ReadOutputAtTime(fname):
    """Distributed blade loading as generated by the HAWC2 output_at_time
    command. From HAWC2 12.3-beta and onwards, there are 7 header columns,
    earlier version only have 3.

    Parameters
    ----------

    fname : str

    header_lnr : int, default=3
        Line number of the header (column names) (1-based counting).
    """
#    data = pd.read_fwf(fname, skiprows=3, header=None)
#    pd.read_table(fname, sep='  ', skiprows=3)
#    data.index.names = cols

    # because the formatting is really weird, we need to sanatize it a bit
    with open(fname, 'r') as f:
        # read the header from line 3
        for k in range(7):
            line = f.readline()
            if line[0:12].lower().replace('#', '').strip() == 'radius_s':
                header_lnr = k + 1
                break
        header = line.replace('\r', '').replace('\n', '')
        cols = [k.strip().replace(' ', '_') for k in header.split('#')[1:]]

    data = np.loadtxt(fname, skiprows=header_lnr)
    return pd.DataFrame(data, columns=cols)


def ReadEigenBody(fname, debug=False):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    Parameters
    ----------

    file_path : str

    file_name : str


    Returns
    -------

    results : DataFrame
        Columns: body, Fd_hz, Fn_hz, log_decr_pct

    """

    # Body data for body number : 3 with the name :nacelle
    # Results:         fd [Hz]       fn [Hz]       log.decr [%]
    # Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02
    FILE = open(fname)
    lines = FILE.readlines()
    FILE.close()

    df_dict = {'Fd_hz': [], 'Fn_hz': [], 'log_decr_pct': [], 'body': []}
    for i, line in enumerate(lines):
        if debug: print('line nr: %5i' % i)
        # identify for which body we will read the data
        if line[:25] == 'Body data for body number':
            body = line.split(':')[2].rstrip().lstrip()
            # remove any annoying characters
            body = body.replace('\n', '').replace('\r', '')
            if debug: print('modes for body: %s' % body)
        # identify mode number and read the eigenfrequencies
        elif line[:8] == 'Mode nr:':
            linelist = line.replace('\n', '').replace('\r', '').split(':')
            # modenr = linelist[1].rstrip().lstrip()
            # text after Mode nr can be empty
            try:
                eigenmodes = linelist[2].rstrip().lstrip().split('   ')
            except IndexError:
                eigenmodes = ['0', '0', '0']

            if debug: print(eigenmodes)
            # in case we have more than 3, remove all the empty ones
            # this can happen when there are NaN values
            if not len(eigenmodes) == 3:
                eigenmodes = linelist[2].rstrip().lstrip().split(' ')
                eigmod = []
                for k in eigenmodes:
                    if len(k) > 1:
                        eigmod.append(k)
                # eigenmodes = eigmod
            else:
                eigmod = eigenmodes
            # remove any trailing spaces for each element
            for k in range(len(eigmod)):
                eigmod[k] = float(eigmod[k])  #.lstrip().rstrip()

            df_dict['body'].append(body)
            df_dict['Fd_hz'].append(eigmod[0])
            df_dict['Fn_hz'].append(eigmod[1])
            df_dict['log_decr_pct'].append(eigmod[2])

    return pd.DataFrame(df_dict)


def ReadEigenStructure(fname, debug=False):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    The file looks as follows:
    #0 Version ID : HAWC2MB 11.3
    #1 ___________________________________________________________________
    #2 Structure eigenanalysis output
    #3 ___________________________________________________________________
    #4 Time : 13:46:59
    #5 Date : 28:11.2012
    #6 ___________________________________________________________________
    #7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #...
    #302  Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    Parameters
    ----------

    file_path : str

    file_name : str

    debug : boolean, default=False

    max_modes : int
        Stop evaluating the result after max_modes number of modes have been
        identified

    Returns
    -------

    modes_arr : ndarray(3,n)
        An ndarray(3,n) holding Fd, Fn [Hz] and the logarithmic damping
        decrement [%] for n different structural eigenmodes

    """

    # 0 Version ID : HAWC2MB 11.3
    # 1 ___________________________________________________________________
    # 2 Structure eigenanalysis output
    # 3 ___________________________________________________________________
    # 4 Time : 13:46:59
    # 5 Date : 28:11.2012
    # 6 ___________________________________________________________________
    # 7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    # 8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #   Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    FILE = open(fname)
    lines = FILE.readlines()
    FILE.close()

    header_lines = 8

    # we now the number of modes by having the number of lines
    nrofmodes = len(lines) - header_lines

    df = pd.DataFrame(np.ndarray((nrofmodes, 3)), dtype=np.float64,
                      columns=['Fd_hz', 'Fn_hz', 'log_decr_pct'])

    for i, line in enumerate(lines):
        # if i > max_modes:
        #     # cut off the unused rest
        #     df.iloc[:,i] = modes_arr[:, :i]
        #     break

        # ignore the header
        if i < header_lines:
            continue

        # split up mode nr from the rest, remove line ending
        parts = line[:-1].split(':')
        #modenr = int(parts[1])
        # get fd, fn and damping, but remove all empty items on the list
        # also cut off line
        df.iloc[i-header_lines,:]=np.float64(misc.remove_items(parts[2].split(' '), ''))

    return df


def ReadStructInertia(fname):

    with open(fname) as f:
        lines = f.readlines()

    marks = []
    for i, line in enumerate(lines):
        if line.startswith('_________') > 0:
            marks.append(i)

    header = ['body_name'] + lines[7].split()[2:]
    data = lines[9:marks[4]]
    bodies = {i:[] for i in header}
    for row in data:
        row_els = row[:-1].split()
        for colname, col  in zip(header, row_els):
            bodies[colname].append(col)

    bodies = pd.DataFrame(bodies)
    for k in header[1:]:
        bodies[k] = bodies[k].astype(float)

    return bodies


class UserWind(object):
    """
    """

    def __init__(self):
        pass

    def __call__(self, z_h, r_blade_tip, a_phi=None, shear_exp=None, nr_hor=3,
                 nr_vert=20, h_ME=500.0, io=None, wdir=None):
        """

        Parameters
        ----------

        z_h : float
            Hub height

        r_blade_tip : float
            Blade tip radius

        a_phi : float, default=None
            :math:`a_{\\varphi} \\approx 0.5` parameter for the modified
            Ekman veer distribution. Values vary between -1.2 and 0.5.

        shear_exp : float, default=None

        nr_vert : int, default=3

        nr_hor : int, default=20

        h_ME : float, default=500
            Modified Ekman parameter. Take roughly 500 for off shore sites,
            1000 for on shore sites.

        io : str or io buffer, default=None
            When specified, the HAWC2 user defined shear input file will be
            written.

        wdir : float, default=None
            A constant veer angle, or yaw angle. Equivalent to setting the
            wind direction. Angle in degrees.

        Returns
        -------

        uu, vv, ww, xx, zz

        """

        x, z = self.create_coords(z_h, r_blade_tip, nr_vert=nr_vert,
                                  nr_hor=nr_hor)
        if a_phi is not None:
            phi_rad = WindProfiles.veer_ekman_mod(z, z_h, h_ME=h_ME, a_phi=a_phi)
            assert len(phi_rad) == nr_vert
        else:
            nr_vert = len(z)
            phi_rad = np.zeros((nr_vert,))
        # add any yaw error on top of
        if wdir is not None:
            # because wdir cw positive, and phi veer ccw positive
            phi_rad -= (wdir*np.pi/180.0)
        u, v, w = self.decompose_veer(phi_rad, nr_hor)
        # when no shear is defined
        if shear_exp is None:
            uu = u
            vv = v
            ww = w
        else:
            # scale the shear on top of the veer
            shear = WindProfiles.powerlaw(z, z_h, shear_exp)
            uu = u*shear[:,np.newaxis]
            vv = v*shear[:,np.newaxis]
            ww = w*shear[:,np.newaxis]
        # and write to a file
        if isinstance(io, str):
            with open(io, 'wb') as fid:
                fid = self.write(fid, uu, vv, ww, x, z)
            self.fid =None
        elif io is not None:
            io = self.write(io, uu, vv, ww, x, z)
            self.fid = io

        return uu, vv, ww, x, z

    def create_coords(self, z_h, r_blade_tip, nr_vert=3, nr_hor=20):
        """
        Utility to create the coordinates of the wind field based on hub heigth
        and blade length. Add 15% to r_blade_tip to make sure horizontal edges
        are defined wide enough.
        """
        # take 15% extra space after the blade tip
        z = np.linspace(0, z_h + r_blade_tip*1.15, nr_vert)
        # along the horizontal, coordinates with 0 at the rotor center
        x = np.linspace(-r_blade_tip*1.15, r_blade_tip*1.15, nr_hor)

        return x, z

    def deltaphi2aphi(self, d_phi, z_h, r_blade_tip, h_ME=500.0):
        """For a given `\\Delta \\varphi` over the rotor diameter, estimate
        the corresponding `a_{\\varphi}`.

        Parameters
        ----------

        `\\Delta \\varphi` : ndarray or float
            Veer angle difference over the rotor plane from lowest to highest
            blade tip position.

        z_h : float
            Hub height in meters.

        r_blade_tip : float
            Blade tip radius/length.

        h_ME : float, default=500.0
            Modified Ekman parameter. For on shore,
            :math:`h_{ME} \\approx 1000`, for off-shore,
            :math:`h_{ME} \\approx 500`

        Returns
        -------

        `a_{\\varphi}` : ndarray or float

        """

        t1 = r_blade_tip * 2.0 * np.exp(-z_h/(h_ME))
        a_phi = d_phi * np.sqrt(h_ME*z_h) / t1
        return a_phi

    def deltaphi2aphi_opt(self, deltaphi, z, z_h, r_blade_tip, h_ME):
        """
        convert delta_phi over a given interval z to a_phi using
        scipy.optimize.fsolve on veer_ekman_mod.

        Parameters
        ----------

        deltaphi : float
            Desired delta phi in rad over interval z[0] at bottom to z[1] at
            the top.


        """

        def func(a_phi, z, z_h, h_ME, deltaphi_target):
            phis = WindProfiles.veer_ekman_mod(z, z_h, h_ME=h_ME, a_phi=a_phi)
            return np.abs(deltaphi_target - (phis[1] - phis[0]))

        args = (z, z_h, h_ME, deltaphi)
        return sp.optimize.fsolve(func, [0], args=args)[0]

    def decompose_veer(self, phi_rad, nr_hor):
        """
        Convert a veer angle into u, v, and w components, ready for the
        HAWC2 user defined veer input file. nr_vert refers to the number of
        vertical grid points.

        Paramters
        ---------

        phi_rad : ndarray(nr_vert)
            veer angle in radians as function of height

        nr_hor : int
            Number of horizontal grid points

        Returns
        -------

        u : ndarray(nr_hor, nr_vert)

        v : ndarray(nr_hor, nr_vert)

        w : ndarray(nr_hor, nr_vert)

        """

        nr_vert = len(phi_rad)
        tan_phi = np.tan(phi_rad)

        # convert veer angles to veer components in v, u. Make sure the
        # normalized wind speed remains 1!
#        u = sympy.Symbol('u')
#        v = sympy.Symbol('v')
#        tan_phi = sympy.Symbol('tan_phi')
#        eq1 = u**2.0 + v**2.0 - 1.0
#        eq2 = (tan_phi*u/v) - 1.0
#        sol = sympy.solvers.solve([eq1, eq2], [u,v], dict=True)
#        # proposed solution is:
#        u2 = np.sqrt(tan_phi**2/(tan_phi**2 + 1.0))/tan_phi
#        v2 = np.sqrt(tan_phi**2/(tan_phi**2 + 1.0))
#        # but that gives the sign switch wrong, simplify/rewrite to:
        u = np.sqrt(1.0/(tan_phi**2 + 1.0))
        v = np.sqrt(1.0/(tan_phi**2 + 1.0))*tan_phi
        # verify they are actually the same but the sign:
#        assert np.allclose(np.abs(u), np.abs(u2))
#        assert np.allclose(np.abs(v), np.abs(v2))

        u_full = u[:, np.newaxis] + np.zeros((3,))[np.newaxis, :]
        v_full = v[:, np.newaxis] + np.zeros((3,))[np.newaxis, :]
        w_full = np.zeros((nr_vert, nr_hor))

        return u_full, v_full, w_full

    def read(self, fname):
        """
        Read a user defined shear input file as used for HAWC2.

        Returns
        -------

        u_comp, v_comp, w_comp, v_coord, w_coord, phi_deg
        """
        # read the header
        with open(fname) as f:
            for i, line in enumerate(f.readlines()):
                if line.strip()[0] != '#':
                    nr_v, nr_w = misc.remove_items(line.split('#')[0].split(), '')
                    nr_hor, nr_vert = int(nr_v), int(nr_w)
                    i_header = i
                    break

        # u,v and w components on 2D grid
        tmp = np.genfromtxt(fname, skip_header=i_header+1, comments='#',
                            max_rows=nr_vert*3)
        if not tmp.shape == (nr_vert*3, nr_hor):
            raise AssertionError('user defined shear input file inconsistent')
        v_comp = tmp[:nr_vert,:]
        u_comp = tmp[nr_vert:nr_vert*2,:]
        w_comp = tmp[nr_vert*2:nr_vert*3,:]

        # coordinates of the 2D grid
        tmp = np.genfromtxt(fname, skip_header=3*(nr_vert+1)+2,
                            max_rows=nr_hor+nr_vert)
        if not tmp.shape == (nr_vert+nr_hor,):
            raise AssertionError('user defined shear input file inconsistent')
        v_coord = tmp[:nr_hor]
        w_coord = tmp[nr_hor:]

        phi_deg = np.arctan(v_comp[:, 0]/u_comp[:, 0])*180.0/np.pi

        return u_comp, v_comp, w_comp, v_coord, w_coord, phi_deg

    def write(self, fid, u, v, w, v_coord, w_coord, fmt_uvw='% 08.05f',
              fmt_coord='% 8.02f'):
        """Write a user defined shear input file for HAWC2.
        """
        nr_hor = len(v_coord)
        nr_vert = len(w_coord)

        try:
            assert u.shape == v.shape
            assert u.shape == w.shape
            assert u.shape[0] == nr_vert
            assert u.shape[1] == nr_hor
        except AssertionError:
            raise ValueError('u, v, w shapes should be consistent with '
                             'nr_hor and nr_vert: u.shape: %s, nr_hor: %i, '
                             'nr_vert: %i' % (str(u.shape), nr_hor, nr_vert))

        fid.write(b'# User defined shear file\n')
        tmp = '%i %i # nr_hor (v), nr_vert (w)\n' % (nr_hor, nr_vert)
        fid.write(tmp.encode())
        h1 = 'normalized with U_mean, nr_hor (v) rows, nr_vert (w) columns'
        fid.write(('# v component, %s\n' % h1).encode())
        np.savetxt(fid, v, fmt=fmt_uvw, delimiter='  ')
        fid.write(('# u component, %s\n' % h1).encode())
        np.savetxt(fid, u, fmt=fmt_uvw, delimiter='  ')
        fid.write(('# w component, %s\n' % h1).encode())
        np.savetxt(fid, w, fmt=fmt_uvw, delimiter='  ')
        h2 = '# v coordinates (along the horizontal, nr_hor, 0 rotor center)'
        fid.write(('%s\n' % h2).encode())
        np.savetxt(fid, v_coord.reshape((v_coord.size, 1)), fmt=fmt_coord)
        h3 = '# w coordinates (zero is at ground level, height, nr_hor)'
        fid.write(('%s\n' % h3).encode())
        np.savetxt(fid, w_coord.reshape((w_coord.size, 1)), fmt=fmt_coord)

        return fid


class WindProfiles(object):

    def logarithmic(z, z_ref, r_0):
        return np.log10(z/r_0)/np.log10(z_ref/r_0)

    def powerlaw(z, z_ref, a):
        profile = np.power(z/z_ref, a)
        # when a negative, make sure we return zero and not inf
        profile[np.isinf(profile)] = 0.0
        return profile

    def veer_ekman_mod(z, z_h, h_ME=500.0, a_phi=0.5):
        """
        Modified Ekman veer profile, as defined by Mark C. Kelly in email on
        10 October 2014 15:10 (RE: veer profile)

        .. math::
            \\varphi(z) - \\varphi(z_H) \\approx a_{\\varphi}
            e^{-\sqrt{z_H/h_{ME}}}
            \\frac{z-z_H}{\sqrt{z_H*h_{ME}}}
            \\left( 1 - \\frac{z-z_H}{2 \sqrt{z_H h_{ME}}}
            - \\frac{z-z_H}{4z_H} \\right)

        where:
        :math:`h_{ME} \\equiv \\frac{\\kappa u_*}{f}`
        and :math:`f = 2 \Omega \sin \\varphi` is the coriolis parameter,
        and :math:`\\kappa = 0.41` as the von Karman constant,
        and :math:`u_\\star = \\sqrt{\\frac{\\tau_w}{\\rho}}` friction velocity.

        For on shore, :math:`h_{ME} \\approx 1000`, for off-shore,
        :math:`h_{ME} \\approx 500`

        :math:`a_{\\varphi} \\approx 0.5`

        Parameters
        ----------

        z : ndarray(n)
            z-coordinates (height) of the grid on which the veer angle should
            be calculated.

        z_h : float
            Hub height in meters.

        :math:`a_{\\varphi}` : default=0.5
            Parameter for the modified Ekman veer distribution. Value varies
            between -1.2 and 0.5.

        Returns
        -------

        phi_rad : ndarray
            Veer angle in radians as function of z.


        """

        t1 = np.exp(-math.sqrt(z_h / h_ME))
        t2 = (z - z_h) / math.sqrt(z_h * h_ME)
        t3 = (1.0 - (z-z_h)/(2.0*math.sqrt(z_h*h_ME)) - (z-z_h)/(4.0*z_h))

        return a_phi * t1 * t2 * t3


if __name__ == '__main__':

    pass
