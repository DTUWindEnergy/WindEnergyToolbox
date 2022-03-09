# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:53:59 2014

@author: dave
"""
__author__ = 'David Verelst'
__license__ = 'GPL'
__version__ = '0.5'

import os
import copy
import struct
import math
from time import time
import codecs
from itertools import chain
import re as re

import numpy as np
import scipy as sp
import scipy.integrate as integrate
import pandas as pd

# misc is part of prepost, which is available on the dtu wind gitlab server:
# https://gitlab.windenergy.dtu.dk/dave/prepost
from wetb.prepost import misc
# wind energy python toolbox, available on the dtu wind redmine server:
# http://vind-redmine.win.dtu.dk/projects/pythontoolbox/repository/show/fatigue_tools
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

    This class is called like a function:
    HawcResultData() will read the specified file upon object initialization.

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
    # ch_df columns, these are created by LoadResults._unified_channel_names
    cols = set(['bearing_name', 'sensortag', 'bodyname', 'chi', 'component',
                'pos', 'coord', 'sensortype', 'radius', 'blade_nr', 'units',
                'output_type', 'io_nr', 'io', 'dll', 'azimuth', 'flap_nr',
                'direction', 'wake_source_nr', 'center', 's', 'srel',
                'radius_actual'])

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

    # TODO: THIS IS STILL A WIP
    def _make_channel_names(self):
        """Give every channel a unique channel name which is (nearly) identical
        to the channel names as defined in the htc output section. Instead
        of spaces, use colon (;) to seperate the different commands.

        THIS IS STILL A WIP

        see also issue #11:
        https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues/11
        """

        index = {}

        names = {'htc_name':[], 'chi':[], 'label':[], 'unit':[], 'index':[],
                 'name':[], 'description':[]}
        constraint_fmts = {'bea1':'constraint;bearing1',
                           'bea2':'constraint;bearing2',
                           'bea3':'constraint;bearing3',
                           'bea4':'constraint;bearing4'}
        # mbdy momentvec tower  1 1 global
        force_fmts = {'F':'mbdy;forcevec;{body};{nodenr:03i};{coord};{comp}',
                      'M':'mbdy;momentvec;{body};{nodenr:03i};{coord};{comp}'}
        state_fmt = 'mbdy;{state};{typ};{body};{elnr:03i};{zrel:01.02f};{coord}'

        wind_coord_map = {'Vx':'1', 'Vy':'2', 'Vz':'3'}
        wind_fmt = 'wind;{typ};{coord};{x};{y};{z};{comp}'

        for ch in range(self.Nch):
            name = self.ch_details[ch, 0]
            name_items = misc.remove_items(name.split(' '), '')

            description = self.ch_details[ch, 2]
            descr_items = misc.remove_items(description.split(' '), '')

            unit = self.ch_details[ch, 1]

            # default names
            htc_name = ' '.join(name_items+descr_items)
            label = ''
            coord = ''
            typ = ''
            elnr = ''
            nodenr = ''
            zrel = ''
            state = ''

            # CONSTRAINTS: BEARINGS
            if name_items[0] in constraint_fmts:
                htc_name = constraint_fmts[name_items[0]] + ';'
                htc_name += (descr_items[0] + ';')
                htc_name += unit

            # MBDY FORCES/MOMENTS
            elif name_items[0][0] in force_fmts:
                comp = name_items[0]
                if comp[0] == 'F':
                    i0 = 1
                else:
                    i0 = 0
                label = description.split('coo: ')[1].split('  ')[1]
                coord = descr_items[i0+5]
                body = descr_items[i0+1][5:]#.replace('Mbdy:', '')
                nodenr = int(descr_items[i0+3])
                htc_name = force_fmts[comp[0]].format(body=body, coord=coord,
                                                      nodenr=nodenr, comp=comp)

            # STATE: POS, VEL, ACC, STATE_ROT
            elif descr_items[0][:5] == 'State':
                if name_items[0] == 'State':
                    i0 = 1
                    state = 'state'
                else:
                    i0 = 0
                    state = 'state_rot'
                typ = name_items[i0+0]
                comp = name_items[i0+1]
                coord = name_items[i0+3]

                body = descr_items[3][5:]#.replace('Mbdy:', '')
                elnr = int(descr_items[5])
                zrel = float(descr_items[6][6:])#.replace('Z-rel:', ''))
                if len(descr_items) > 8:
                    label = ' '.join(descr_items[9:])
                htc_name = state_fmt.format(typ=typ, body=body, elnr=elnr,
                                            zrel=zrel, coord=coord,
                                            state=state)

            # WINDSPEED
            elif description[:9] == 'Free wind':
                if descr_items[4] == 'gl.':
                    coord = '1' # global
                else:
                    coord = '2' # non-rotating rotor coordinates

                try:
                    comp = wind_coord_map[descr_items[3][:-1]]
                    typ = 'free_wind'
                except KeyError:
                    comp = descr_items[3]
                    typ = 'free_wind_hor'

                tmp = description.split('pos')[1]
                x, y, z = tmp.split(',')
                # z might hold a label....
                z_items  = z.split('  ')
                if len(z_items) > 1:
                    label = '  '.join(z_items[1:])
                    z = z_items[0]
                x, y, z = x.strip(), y.strip(), z.strip()

                htc_name = wind_fmt.format(typ=typ, coord=coord, x=x, y=y, z=z,
                                           comp=comp)


            names['htc_name'].append(htc_name)
            names['chi'].append(ch)
            # this is the Channel column from the sel file, so the unique index
            # which is dependent on the order of the channels
            names['index'].append(ch+1)
            names['unit'].append(unit)
            names['name'].append(name)
            names['description'].append(description)
            names['label'].append(label)
            names['state'].append(state)
            names['type'].append(typ)
            names['comp'].append(comp)
            names['coord'].append(coord)
            names['elnr'].append(coord)
            names['nodenr'].append(coord)
            names['zrel'].append(coord)
            index[name] = ch

        return names, index

    def _unified_channel_names(self):
        """
        Make certain channels independent from their index.

        The unified channel dictionary ch_dict holds consequently named
        channels as the key, and the all information is stored in the value
        as another dictionary.

        The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            node numbers start with 0 at the root
            element numbers start with 1 at the root
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos']
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

        For many of the aero sensors:
            'Cl', 'Cd', 'Alfa', 'Vrel'
            key:
                sensortype-blade_nr-pos
                Cl-1-0.01
            value:
                ch_dict[tag]['sensortype']
                ch_dict[tag]['blade_nr']
                ch_dict[tag]['pos']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']
        """
        # save them in a dictionary, use the new coherent naming structure
        # as the key, and as value again a dict that hols all the different
        # classifications: (chi, channel nr), (coord, coord), ...
        self.ch_dict = dict()

        # some channel ID's are unique, use them
        ch_unique = set(['Omega', 'Ae rot. torque', 'Ae rot. power',
                         'Ae rot. thrust', 'Time', 'Azi  1'])
        ch_aero = set(['Cl', 'Cd', 'Cm', 'Alfa', 'Vrel', 'Tors_e', 'Alfa',
                       'Lift', 'Drag'])
        ch_aerogrid = set(['a_grid', 'am_grid', 'CT', 'CQ'])

        # also safe as df
#        cols = set(['bearing_name', 'sensortag', 'bodyname', 'chi',
#                    'component', 'pos', 'coord', 'sensortype', 'radius',
#                    'blade_nr', 'units', 'output_type', 'io_nr', 'io', 'dll',
#                    'azimuth', 'flap_nr'])
        df_dict = {col: [] for col in self.cols}
        df_dict['unique_ch_name'] = []

        # -----------------------------------------------------------------
        # REGEXes
        # -----------------------------------------------------------------

        # ESYS output: ESYS line3 SENSOR           66
        re_esys = re.compile(r'ESYS (\w+) SENSOR\s*(\d*)')
        # FORCE fext_damp   1
        re_force = re.compile(r'FORCE (\w+) \s*(\d*)')
        # scan through all channels and see which can be converted
        # to sensible unified name
        for ch in range(self.Nch):

            items_ch0 = self.ch_details[ch, 0].split()
            items_ch2 = self.ch_details[ch, 2].split()

            dll = False

            # be carefull, identify only on the starting characters, because
            # the signal tag can hold random text that in some cases might
            # trigger a false positive

            # -----------------------------------------------------------------
            # check for all the unique channel descriptions
            if self.ch_details[ch,0].strip() in ch_unique:
                tag = self.ch_details[ch, 0].strip()
                channelinfo = {}
                channelinfo['units'] = self.ch_details[ch, 1]
                channelinfo['sensortag'] = self.ch_details[ch, 2]
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # or in the long description:
            #    0          1        2      3  4    5     6 and up
            # MomentMz Mbdy:blade nodenr:   5 coo: blade  TAG TEXT
            elif self.ch_details[ch, 2].startswith('MomentM'):
                coord = items_ch2[5]
                bodyname = items_ch2[1].replace('Mbdy:', '')
                # set nodenr to sortable way, include leading zeros
                # node numbers start with 0 at the root
                nodenr = '%03i' % int(items_ch2[3])
                # skip the attached the component
                # sensortype = items[0][:-2]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'momentvec'
                component = items_ch2[0][-1:len(items_ch2[0])]
                # the tag only exists if defined
                if len(items_ch2) > 6:
                    sensortag = ' '.join(items_ch2[6:])
                else:
                    sensortag = ''

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord, bodyname, pos, sensortype, component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            #   0    1      2        3       4  5     6     7 and up
            # Force  Fx Mbdy:blade nodenr:   2 coo: blade  TAG TEXT
            elif self.ch_details[ch, 2].startswith('Force  F'):
                coord = items_ch2[6]
                bodyname = items_ch2[2].replace('Mbdy:', '')
                nodenr = '%03i' % int(items_ch2[4])
                # skipe the attached the component
                # sensortype = items[0]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'forcevec'
                component = items_ch2[1][1]
                if len(items_ch2) > 7:
                    sensortag = ' '.join(items_ch2[7:])
                else:
                    sensortag = ''

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord, bodyname, pos, sensortype, component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            #        0    1      2        3       4    5     6     7    8           9 and up
            # Force_intp  Fz Mbdy:blade1 s=  11.87[m] s/S=   0.95 coo: local_aero center:default
            # Moment_intp  Mx Mbdy:blade1 s=  11.87[m] s/S=   0.95 coo: local_aero center:default
            elif items_ch2[0].endswith('_intp'):

                sensortype = 'forcemomentvec_interp'

                coord = items_ch2[8]
                bodyname = items_ch2[2].replace('Mbdy:', '')
                s = items_ch2[4].replace('[m]', '')
                srel = items_ch2[6]
                center = items_ch2[9].split(':')[1]
                component = items_ch2[1]

                if len(items_ch2) > 9:
                    sensortag = ' '.join(items_ch2[10:])
                else:
                    sensortag = ''

                # and tag it
                pos = 's-%s' % (s)
                tag = f'{sensortype}-{bodyname}-{center}-{coord}-{s}-{component}'
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['s'] = float(s)
                channelinfo['srel'] = float(srel)
                channelinfo['sensortype'] = sensortype
                # channelinfo['output_type'] = output_type
                channelinfo['component'] = component
                channelinfo['center'] = center
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            # ELEMENT STATES: pos, vel, acc, rot, ang
            #   0    1  2      3       4      5   6         7    8
            # State pos x  Mbdy:blade E-nr:   1 Z-rel:0.00 coo: blade
            #   0           1     2    3        4    5   6         7     8     9+
            # State_rot proj_ang tx Mbdy:bname E-nr: 1 Z-rel:0.00 coo: cname  label
            # State_rot omegadot tz Mbdy:bname E-nr: 1 Z-rel:1.00 coo: cname  label
            elif self.ch_details[ch,2].startswith('State'):
#                 or self.ch_details[ch,0].startswith('euler') \
#                 or self.ch_details[ch,0].startswith('ax') \
#                 or self.ch_details[ch,0].startswith('omega') \
#                 or self.ch_details[ch,0].startswith('proj'):
                coord = items_ch2[8]
                bodyname = items_ch2[3].replace('Mbdy:', '')
                # element numbers start with 1 at the root
                elementnr = '%03i' % int(items_ch2[5])
                zrel = '%04.2f' % float(items_ch2[6].replace('Z-rel:', ''))
                # skip the attached the component
                #sensortype = ''.join(items[0:2])
                # or give the sensor type the same name as in HAWC2
                tmp = self.ch_details[ch, 0].split(' ')
                sensortype = tmp[0]
                if sensortype.startswith('State'):
                    sensortype += ' ' + tmp[1]
                component = items_ch2[2]
                if len(items_ch2) > 8:
                    sensortag = ' '.join(items_ch2[9:])
                else:
                    sensortag = ''

                # and tag it
                pos = 'elem-%s-zrel-%s' % (elementnr, zrel)
                tagitems = (coord, bodyname, pos, sensortype, component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            # statevec_new
            #   0     1          2      3  4           5     6       7
            # elastic Deflection blade1 Dx Mbdy:blade1 s=   0.00[m] s/S=
            # 8     9     10     11
            # 0.00 coo: blade1 center:c2def
            # note that: 2 and 10 are the same
            elif items_ch2[0] == 'elastic' or items_ch2[0] == 'absolute':
                output_type = ' '.join(items_ch2[0:2])
                bodyname = items_ch2[4].replace('Mbdy:', '')
                s = '%06.02f' % float(items_ch2[6].replace('[m]', ''))
                srel = '%04.02f' % float(items_ch2[8])
                coord = items_ch2[10]
                center = items_ch2[11].split(':')[1]
                sensortype = 'statevec_new'

                component = items_ch0[0]

                if len(items_ch2) > 12:
                    sensortag = ' '.join(items_ch2[12:])
                else:
                    sensortag = ''

                # and tag it, allmost the same as in htc file here
                tagitems = (sensortype, bodyname, center, coord, items_ch2[0],
                            s, component)
                tag = '-'.join(['%s']*7) % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['s'] = float(s)
                channelinfo['srel'] = float(srel)
                channelinfo['sensortype'] = sensortype
                channelinfo['output_type'] = output_type
                channelinfo['component'] = component
                channelinfo['center'] = center
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            # DLL CONTROL I/O
            # there are two scenario's on how the channel description is formed
            # the channel id is always the same though
            # id for all three cases:
            #          DLL out  1:  3
            #          DLL inp  2:  3
            # description case 1 ("dll type2_dll b2h2 inpvec 30" in htc output)
            #               0         1    2   3     4+
            #          yaw_control outvec  3  yaw_c input reference angle
            # description case 2 ("dll inpvec 2 1" in htc output):
            #           0  1 2     3  4  5  6+
            #          DLL : 2 inpvec :  4  mgen hss
            # description case 3
            #           0         1     2       4
            #          hawc_dll :echo outvec :  1
            elif self.ch_details[ch, 0].startswith('DLL'):
                # case 3
                if items_ch2[0] == 'hawc_dll':
                    # hawc_dll named case (case 3) is polluted with colons
                    dll = items_ch2[1].replace(':', '')
                    io = items_ch2[2]
                    io_nr = items_ch2[4]
                    tag = 'DLL-%s-%s-%s' % (dll, io, io_nr)
                    sensortag = ''
                # case 2: no reference to dll name
                elif self.ch_details[ch,2].startswith('DLL'):
                    dll = items_ch2[2]
                    io = items_ch2[3]
                    io_nr = items_ch2[5]
                    sensortag = ' '.join(items_ch2[6:])
                    # and tag it
                    tag = 'DLL-%s-%s-%s' % (dll,io,io_nr)
                # case 1: type2 dll name is given
                else:
                    dll = items_ch2[0]
                    io = items_ch2[1]
                    io_nr = items_ch2[2]
                    sensortag = ' '.join(items_ch2[3:])
                    tag = 'DLL-%s-%s-%s' % (dll, io, io_nr)

                # save all info in the dict
                channelinfo = {}
                channelinfo['dll'] = dll
                channelinfo['io'] = io
                channelinfo['io_nr'] = io_nr
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]
                channelinfo['sensortype'] = 'dll-io'

            # -----------------------------------------------------------------
            # BEARING OUTPUS
            # bea1 angle_speed       rpm      shaft_nacelle angle speed
            elif self.ch_details[ch, 0].startswith('bea'):
                output_type = self.ch_details[ch, 0].split(' ')[1]
                bearing_name = items_ch2[0]
                units = self.ch_details[ch, 1]
                # there is no label option for the bearing output

                # and tag it
                tag = 'bearing-%s-%s-%s' % (bearing_name, output_type, units)
                # save all info in the dict
                channelinfo = {}
                channelinfo['bearing_name'] = bearing_name
                channelinfo['output_type'] = output_type
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # AS DEFINED IN: ch_aero
            # AERO CL, CD, CM, VREL, ALFA, LIFT, DRAG, etc
            # Cl, R=  0.5     deg      Cl of blade  1 at radius   0.49
            # Azi  1          deg      Azimuth of blade  1
            #
            # ch_details[ch, 2]:
            # Angle of attack of blade   1 at radius   8.59 FOLLOWD BY USER LABEL
            #
            # NOTE THAT RADIUS FROM ch_details[ch, 0] REFERS TO THE RADIUS
            # YOU ASKED FOR, AND ch_details[ch, 2] IS WHAT YOU GET, which is
            # still based on a mean radius (deflections change the game)
            elif self.ch_details[ch, 0].split(',')[0] in ch_aero:
                sensortype = self.ch_details[ch, 0].split(',')[0]

                # sometimes the units for aero sensors are wrong!
                units = self.ch_details[ch, 1]
                # there is no label option

                # Blade number is identified as the first integer in the string
                # blade_nr = re.search(r'\d+', self.ch_details[ch, 2]).group()
                # blade_nr = int(blade_nr)

                # actual radius
                rq = r'\.*of blade\s*(\d) at radius\s*([-+]?\d*\.\d+|\d+)'
                s = self.ch_details[ch, 2]
                blade_nr, radius_actual = re.findall(rq, s)[0]
                blade_nr = int(blade_nr)

                # radius what you asked for, identified as the last float in the string
                s = self.ch_details[ch, 0]
                radius = float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])

                # and tag it
                tag = '%s-%s-%s' % (sensortype, blade_nr, radius)
                # save all info in the dict
                channelinfo = {}
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = float(radius)
                channelinfo['radius_actual'] = float(radius_actual)
                channelinfo['blade_nr'] = blade_nr
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # for the induction grid over the rotor
            # a_grid, azi    0.00 r   1.74
            elif self.ch_details[ch, 0].split(',')[0] in ch_aerogrid:
                items_ = self.ch_details[ch, 0].split(',')
                sensortype = items_[0]
                items2 = items_[1].split(' ')
                items2 = misc.remove_items(items2, '')
                azi = items2[1]
                # radius what you asked for
                radius = items2[3]
                units = self.ch_details[ch, 1]
                # and tag it
                tag = '%s-azi-%s-r-%s' % (sensortype,azi,radius)
                # save all info in the dict
                channelinfo = {}
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = float(radius)
                channelinfo['azimuth'] = float(azi)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # INDUCTION AT THE BLADE
            # 0: Induc. Vz, rpco, R=  1.4
            # 1: m/s
            # 2: Induced wsp Vz of blade  1 at radius   1.37, RP. coo.
            # Induc. Vx, locco, R=  1.4
            #    Induced wsp Vx of blade  1 at radius   1.37, local ae coo.
            # Induc. Vy, blco, R=  1.4
            #    Induced wsp Vy of blade  1 at radius   1.37, local bl coo.
            # Induc. Vz, glco, R=  1.4
            #    Induced wsp Vz of blade  1 at radius   1.37, global coo.
            # Induc. Vx, rpco, R=  8.4
            #    Induced wsp Vx of blade  1 at radius   8.43, RP. coo.
            elif self.ch_details[ch, 0].strip()[:5] == 'Induc':

                coord = self.ch_details[ch, 2].split(', ')[1].strip()
                blade_nr = int(items_ch2[5])

                # radius what you get
                #  radius = float(items[8].replace(',', ''))
                # radius what you asked for, identified as the last float in the string
#                s = self.ch_details[ch, 2]
#                radius = float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])
                radius = float(items_ch2[8][:-1])

                component = items_ch2[2]
                units = self.ch_details[ch, 1]

                # and tag it
                rpl = (coord, blade_nr, component, radius)
                tag = 'induc-%s-blade-%1i-%s-r-%03.01f' % rpl
                # save all info in the dict
                channelinfo = {}
                channelinfo['blade_nr'] = blade_nr
                channelinfo['sensortype'] = 'induction'
                channelinfo['radius'] = radius
                channelinfo['coord'] = coord
                channelinfo['component'] = component
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # MORE AERO SENSORS
            # Ae intfrc Fx, rpco, R=  0.0
            #     Aero int. force Fx of blade  1 at radius   0.00, RP coo.
            # Ae secfrc Fy, R= 25.0
            #     Aero force  Fy of blade  1 at radius  24.11
            # Ae pos x, glco, R= 88.2
            #     Aero position x of blade  1 at radius  88.17, global coo.
            elif self.ch_details[ch, 0].strip()[:2] == 'Ae':
                units = self.ch_details[ch, 1]
                # Blade number is identified as the first integer in the string
                blade_nr = re.search(r'\d+', self.ch_details[ch, 2]).group()
                blade_nr = int(blade_nr)
                # radius what you get
                tmp = self.ch_details[ch, 2].split('radius ')[1].strip()
                tmp = tmp.split(',')
                # radius = float(tmp[0])
                # radius what you asked for, identified as the last float in the string
                s = self.ch_details[ch, 2]
                radius = float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])

                if len(tmp) > 1:
                    coord = tmp[1].strip()
                else:
                    coord = 'aero'

                sensortype = items_ch0[1]
                component = items_ch0[2].replace(',', '')

                # save all info in the dict
                channelinfo = {}
                channelinfo['blade_nr'] = blade_nr
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = radius
                channelinfo['coord'] = coord
                channelinfo['component'] = component
                channelinfo['units'] = units
                channelinfo['chi'] = ch

                rpl = (coord, blade_nr, sensortype, component, radius)
                tag = 'aero-%s-blade-%1i-%s-%s-r-%03.01f' % rpl

            # TODO: wind speed
            # some spaces have been trimmed here
            # WSP gl. coo.,Vy          m/s
            # // Free wind speed Vy, gl. coo, of gl. pos   0.00,  0.00,  -2.31
            # WSP gl. coo.,Vdir_hor          deg
            # Free wind speed Vdir_hor, gl. coo, of gl. pos  0.00,  0.00, -2.31

            # -----------------------------------------------------------------
            # WATER SURFACE gl. coo, at gl. coo, x,y=   0.00,   0.00
            elif self.ch_details[ch, 2].startswith('Water'):
                units = self.ch_details[ch, 1]

                # but remove the comma
                x = items_ch2[-2][:-1]
                y = items_ch2[-1]

                # and tag it
                tag = 'watersurface-global-%s-%s' % (x, y)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = 'global'
                channelinfo['pos'] = (float(x), float(y))
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # WIND SPEED
            elif self.ch_details[ch, 2].startswith('Free wind speed'):
                units = self.ch_details[ch, 1]
                direction = self.ch_details[ch, 0].split(',')[1]
                # WSP gl. coo.,Vx
                # Free wind speed Vx, gl. coo, of gl. pos    0.00,   0.00,  -6.00  LABEL
                if self.ch_details[ch, 2].startswith('Free '):
                    tmp = self.ch_details[ch, 2].split('pos')[1]
                    x, y, z = tmp.split(',')
                    x, y, z = x.strip(), y.strip(), z.strip()
                    tmp = z.split('  ')
                    sensortag = ''
                    if len(tmp) == 2:
                        z, sensortag = tmp
                    elif len(tmp) == 1:
                        z = tmp[0]
                    pos = (float(x), float(y), float(z))
                    posstr = '%s-%s-%s' % (x, y, z)
                    coord = 'global'
                else:
                    pos = items_ch2[6]
                    posstr = pos
                    coord = items_ch2[0].lower()
                    if len(items_ch2) > 6:
                        sensortag = ' '.join(items_ch2[7:])

                # and tag it
                tag = 'windspeed-%s-%s-%s' % (coord, direction, posstr)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = 'global'
                channelinfo['pos'] = pos
                channelinfo['units'] = units
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                # FIXME: direction is the same as component, right?
                channelinfo['direction'] = direction
                channelinfo['sensortype'] = 'wsp-global'

            # WIND SPEED AT BLADE
            # 0: WSP Vx, glco, R= 61.5
            # 2: Wind speed Vx of blade  1 at radius  61.52, global coo.
            elif self.ch_details[ch, 0].startswith('WSP V'):
                units = self.ch_details[ch, 1].strip()
                tmp = self.ch_details[ch, 0].split(' ')[1].strip()
                direction = tmp.replace(',', '')
                coord = self.ch_details[ch, 2].split(',')[1].split()[0]
                # Blade number is identified as the first integer in the string
                blade_nr = re.search(r'\d+', self.ch_details[ch, 2]).group()
                blade_nr = int(blade_nr)

                # radius what you get
                # radius = self.ch_details[ch, 2].split('radius')[1].split(',')[0]
                # radius = radius.strip()
                # radius what you asked for, identified as the last float in the string
                s=self.ch_details[ch, 2]
                radius=float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])

                # and tag it
                rpl = (direction, blade_nr, radius, coord)
                tag = 'wsp-blade-%s-%s-%s-%s' % rpl
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                # FIXME: direction is the same as component, right?
                channelinfo['direction'] = direction
                channelinfo['blade_nr'] = blade_nr
                channelinfo['radius'] = float(radius)
                channelinfo['units'] = units
                channelinfo['chi'] = ch
                channelinfo['sensortype'] = 'wsp-blade'

            # FLAP ANGLE
            # 2: Flap angle for blade  3 flap number  1
            elif self.ch_details[ch, 0][:7] == 'setbeta':
                units = self.ch_details[ch, 1].strip()
                # Blade number is identified as the first integer in the string
                blade_nr = re.search(r'\d+', self.ch_details[ch, 2]).group()
                blade_nr = int(blade_nr)
                flap_nr = self.ch_details[ch, 2].split(' ')[-1].strip()

                # and tag it
                tag = 'setbeta-bladenr-%s-flapnr-%s' % (blade_nr, flap_nr)
                # save all info in the dict
                channelinfo = {}
                channelinfo['flap_nr'] = int(flap_nr)
                channelinfo['blade_nr'] = blade_nr
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # harmonic channel output
            # Harmonic
            # Harmonic sinus function
            elif self.ch_details[ch, 0][:7] == 'Harmoni':

                func_name = ' '.join(self.ch_details[ch, 1].split(' ')[1:])

                channelinfo = {}
                channelinfo['output_type'] = func_name
                channelinfo['sensortype'] = 'harmonic'
                channelinfo['chi'] = ch

                base = self.ch_details[ch,2].strip().lower().replace(' ', '_')
                tag = base + '_0'
                if tag in self.ch_dict:
                    tag_nr = int(tag.split('_')[-1]) + 1
                    tag = base + '_%i' % tag_nr

            elif self.ch_details[ch, 0][:6] == 'a_norm':
                channelinfo = {}
                channelinfo['chi'] = ch
                channelinfo['units'] = self.ch_details[ch, 1].strip()
                channelinfo['sensortype'] = 'aero'
                tag = 'aero-induc_a_norm'

            # wake   1 gl. pos pos_z  //  Wake pos_z of source   1, Met. coo.
            elif self.ch_details[ch, 0][:4] == 'wake':
                wake_nr = re.search(r'\d+', self.ch_details[ch,0]).group()
                comp = re.search(r'pos_([xyz])', self.ch_details[ch,0]).group(1)

                channelinfo = {}
                channelinfo['output_type'] = 'wake_pos'
                channelinfo['sensortype'] = 'wind_wake'
                channelinfo['component'] = comp
                channelinfo['units'] = self.ch_details[ch, 1].strip()
                channelinfo['chi'] = ch
                channelinfo['wake_source_nr'] = int(wake_nr)
                channelinfo['coord'] = 'met'

                tag = 'wind_wake-wake_pos_%s_%s' % (comp, wake_nr)

            # ESYS line1 SENSOR            1
            elif self.ch_details[ch, 2][:4] == 'ESYS':
                # body = re.findall(regex, self.ch_details[ch, 2])
                body, outnr = re_esys.match(self.ch_details[ch, 2]).groups()

                channelinfo = {}
                channelinfo['output_type'] = 'esys'
                channelinfo['sensortype'] = 'esys'
                channelinfo['io_nr'] = int(outnr)
                channelinfo['units'] = self.ch_details[ch, 1].strip()
                channelinfo['chi'] = ch

                tag = 'esys-%s-%s' % (body, outnr)

            elif self.ch_details[ch, 2][:4] == 'FORC':
                # body = re.findall(regex, self.ch_details[ch, 2])
                dllname, outnr = re_force.match(self.ch_details[ch, 2]).groups()

                channelinfo = {}
                channelinfo['output_type'] = 'force-dll'
                channelinfo['sensortype'] = 'force-dll'
                channelinfo['io_nr'] = int(outnr)
                channelinfo['units'] = self.ch_details[ch, 1].strip()
                channelinfo['chi'] = ch

                tag = 'force-%s-%s' % (dllname, outnr)

            # -----------------------------------------------------------------
            # If all this fails, just combine channel name and description
            else:
                tag = '-'.join(self.ch_details[ch,:3].tolist())
                channelinfo = {}
                channelinfo['chi'] = ch
                channelinfo['units'] = self.ch_details[ch, 1].strip()

            # -----------------------------------------------------------------
            # add a v_XXX tag in case the channel already exists
            if tag in self.ch_dict:
                jj = 1
                while True:
                    tag_new = tag + '_v%i' % jj
                    if tag_new in self.ch_dict:
                        jj += 1
                    else:
                        tag = tag_new
                        break

            self.ch_dict[tag] = copy.copy(channelinfo)

            # -----------------------------------------------------------------
            # save in for DataFrame format
            cols_ch = set(channelinfo.keys())
            for col in cols_ch:
                df_dict[col].append(channelinfo[col])
            # the remainder columns we have not had yet. Fill in blank
            for col in (self.cols - cols_ch):
                df_dict[col].append('')
            df_dict['unique_ch_name'].append(tag)

        self.ch_df = pd.DataFrame(df_dict)
        self.ch_df.set_index('chi', inplace=True)


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
        stats['int'] = integrate.trapz(sig[i0:i1, :], x=sig[i0:i1, 0], axis=0)
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
        statsdel['int'] = integrate.trapz(datasel, x=time, axis=0)
        statsdel['intabs'] = integrate.trapz(np.abs(datasel), x=time, axis=0)

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


class Turbulence(object):

    def __init__(self):

        pass

    def read_hawc2(self, fpath, shape):
        """
        Read the HAWC2 turbulence format
        """

        fid = open(fpath, 'rb')
        tmp = np.fromfile(fid, 'float32', shape[0]*shape[1]*shape[2])
        turb = np.reshape(tmp, shape)

        return turb

    def read_bladed(self, fpath, basename):

        fid = open(fpath + basename + '.wnd', 'rb')
        R1 = struct.unpack('h', fid.read(2))[0]
        R2 = struct.unpack('h', fid.read(2))[0]
        turb = struct.unpack('i', fid.read(4))[0]
        lat = struct.unpack('f', fid.read(4))[0]
        rough = struct.unpack('f', fid.read(4))[0]
        refh = struct.unpack('f', fid.read(4))[0]
        longti = struct.unpack('f', fid.read(4))[0]
        latti = struct.unpack('f', fid.read(4))[0]
        vertti = struct.unpack('f', fid.read(4))[0]
        dv = struct.unpack('f', fid.read(4))[0]
        dw = struct.unpack('f', fid.read(4))[0]
        du = struct.unpack('f', fid.read(4))[0]
        halfalong = struct.unpack('i', fid.read(4))[0]
        mean_ws = struct.unpack('f', fid.read(4))[0]
        VertLongComp = struct.unpack('f', fid.read(4))[0]
        LatLongComp = struct.unpack('f', fid.read(4))[0]
        LongLongComp = struct.unpack('f', fid.read(4))[0]
        Int = struct.unpack('i', fid.read(4))[0]
        seed = struct.unpack('i', fid.read(4))[0]
        VertGpNum = struct.unpack('i', fid.read(4))[0]
        LatGpNum = struct.unpack('i', fid.read(4))[0]
        VertLatComp = struct.unpack('f', fid.read(4))[0]
        LatLatComp = struct.unpack('f', fid.read(4))[0]
        LongLatComp = struct.unpack('f', fid.read(4))[0]
        VertVertComp = struct.unpack('f', fid.read(4))[0]
        LatVertComp = struct.unpack('f', fid.read(4))[0]
        LongVertComp = struct.unpack('f', fid.read(4))[0]

        points = np.fromfile(fid, 'int16', 2*halfalong*VertGpNum*LatGpNum*3)
        fid.close()
        return points

    def convert2bladed(self, fpath, basename, shape=(4096,32,32)):
        """
        Convert turbulence box to BLADED format
        """

        u = self.read_hawc2(fpath + basename + 'u.bin', shape)
        v = self.read_hawc2(fpath + basename + 'v.bin', shape)
        w = self.read_hawc2(fpath + basename + 'w.bin', shape)

        # mean velocity components at the center of the box
        v1, v2 = (shape[1]/2)-1, shape[1]/2
        w1, w2 = (shape[2]/2)-1, shape[2]/2
        ucent = (u[:, v1, w1] + u[:, v1, w2] + u[:, v2, w1] + u[:, v2, w2]) / 4.0
        vcent = (v[:, v1, w1] + v[:, v1, w2] + v[:, v2, w1] + v[:, v2, w2]) / 4.0
        wcent = (w[:, v1, w1] + w[:, v1, w2] + w[:, v2, w1] + w[:, v2, w2]) / 4.0

        # FIXME: where is this range 351:7374 coming from?? The original script
        # considered a box of lenght 8192
        umean = np.mean(ucent[351:7374])
        vmean = np.mean(vcent[351:7374])
        wmean = np.mean(wcent[351:7374])

        ustd = np.std(ucent[351:7374])
        vstd = np.std(vcent[351:7374])
        wstd = np.std(wcent[351:7374])

        # gives a slight different outcome, but that is that significant?
#        umean = np.mean(u[351:7374,15:17,15:17])
#        vmean = np.mean(v[351:7374,15:17,15:17])
#        wmean = np.mean(w[351:7374,15:17,15:17])

        # this is wrong since we want the std on the center point
#        ustd = np.std(u[351:7374,15:17,15:17])
#        vstd = np.std(v[351:7374,15:17,15:17])
#        wstd = np.std(w[351:7374,15:17,15:17])

        iu = np.zeros(shape)
        iv = np.zeros(shape)
        iw = np.zeros(shape)

        iu[:, :, :] = (u - umean)/ustd*1000.0
        iv[:, :, :] = (v - vmean)/vstd*1000.0
        iw[:, :, :] = (w - wmean)/wstd*1000.0

        # because MATLAB and Octave do a round when casting from float to int,
        # and Python does a floor, we have to round first
        np.around(iu, decimals=0, out=iu)
        np.around(iv, decimals=0, out=iv)
        np.around(iw, decimals=0, out=iw)

        return iu.astype(np.int16), iv.astype(np.int16), iw.astype(np.int16)

    def write_bladed(self, fpath, basename, shape):
        """
        Write turbulence BLADED file
        """
        # TODO: get these parameters from a HAWC2 input file
        seed = 6
        mean_ws = 11.4
        turb = 3
        R1 = -99
        R2 = 4

        du = 0.974121094
        dv = 4.6875
        dw = 4.6875

        longti = 14
        latti = 9.8
        vertti = 7

        iu, iv, iw = self.convert2bladed(fpath, basename, shape=shape)

        fid = open(fpath + basename + '.wnd', 'wb')
        fid.write(struct.pack('h', R1))  # R1
        fid.write(struct.pack('h', R2))  # R2
        fid.write(struct.pack('i', turb))  # Turb
        fid.write(struct.pack('f', 999))  # Lat
        fid.write(struct.pack('f', 999))  # rough
        fid.write(struct.pack('f', 999))  # refh
        fid.write(struct.pack('f', longti))  # LongTi
        fid.write(struct.pack('f', latti))  # LatTi
        fid.write(struct.pack('f', vertti))  # VertTi
        fid.write(struct.pack('f', dv))  # VertGpSpace
        fid.write(struct.pack('f', dw))  # LatGpSpace
        fid.write(struct.pack('f', du))  # LongGpSpace
        fid.write(struct.pack('i', shape[0]/2))  # HalfAlong
        fid.write(struct.pack('f', mean_ws))  # meanWS
        fid.write(struct.pack('f', 999.))  # VertLongComp
        fid.write(struct.pack('f', 999.))  # LatLongComp
        fid.write(struct.pack('f', 999.))  # LongLongComp
        fid.write(struct.pack('i', 999))  # Int
        fid.write(struct.pack('i', seed))  # Seed
        fid.write(struct.pack('i', shape[1]))  # VertGpNum
        fid.write(struct.pack('i', shape[2]))  # LatGpNum
        fid.write(struct.pack('f', 999))  # VertLatComp
        fid.write(struct.pack('f', 999))  # LatLatComp
        fid.write(struct.pack('f', 999))  # LongLatComp
        fid.write(struct.pack('f', 999))  # VertVertComp
        fid.write(struct.pack('f', 999))  # LatVertComp
        fid.write(struct.pack('f', 999))  # LongVertComp
#        fid.flush()

#        bladed2 = np.ndarray((shape[0], shape[2], shape[1], 3), dtype=np.int16)
#        for i in xrange(shape[0]):
#            for k in xrange(shape[1]):
#                for j in xrange(shape[2]):
#                    fid.write(struct.pack('i', iu[i, shape[1]-j-1, k]))
#                    fid.write(struct.pack('i', iv[i, shape[1]-j-1, k]))
#                    fid.write(struct.pack('i', iw[i, shape[1]-j-1, k]))
#                    bladed2[i,k,j,0] = iu[i, shape[1]-j-1, k]
#                    bladed2[i,k,j,1] = iv[i, shape[1]-j-1, k]
#                    bladed2[i,k,j,2] = iw[i, shape[1]-j-1, k]

        # re-arrange array for bladed format
        bladed = np.ndarray((shape[0], shape[2], shape[1], 3), dtype=np.int16)
        bladed[:, :, :, 0] = iu[:, ::-1, :]
        bladed[:, :, :, 1] = iv[:, ::-1, :]
        bladed[:, :, :, 2] = iw[:, ::-1, :]
        bladed_swap_view = bladed.swapaxes(1,2)
        bladed_swap_view.tofile(fid, format='%int16')

        fid.flush()
        fid.close()


class Bladed(object):

    def __init__(self):
        """
        Some BLADED results I have seen are just weird text files. Convert
        them to a more convienent format.

        path/to/file
        channel 1 description
        col a name/unit col b name/unit
        a0 b0
        a1 b1
        ...
        path/to/file
        channel 2 description
        col a name/unit col b name/unit
        ...
        """
        pass

    def infer_format(self, lines):
        """
        Figure out how many channels and time steps are included
        """
        count = 1
        for line in lines[1:]:
            if line == lines[0]:
                break
            count += 1
        iters = count - 3
        chans = len(lines) / (iters + 3)
        return int(chans), int(iters)

    def read(self, fname, chans=None, iters=None, enc='cp1252'):
        """
        Parameters
        ----------

        fname : str

        chans : int, default=None

        iters : int, default=None

        enc : str, default='cp1252'
            character encoding of the source file. Usually BLADED is used on
            windows so Western-European windows encoding is a safe bet.
        """

        with codecs.open(fname, 'r', enc) as f:
            lines = f.readlines()
        nrl = len(lines)
        if chans is None and iters is None:
            chans, iters = self.infer_format(lines)
        if iters is not None:
            chans = int(nrl / (iters + 3))
        if chans is not None:
            iters = int((nrl / chans) - 3)
#        file_head = [ [k[:-2],0] for k in lines[0:nrl:iters+3] ]
#        chan_head = [ [k[:-2],0] for k in lines[1:nrl:iters+3] ]
#        cols_head = [ k.split('\t')[:2] for k in lines[2:nrl:iters+3] ]

        data = {}
        for k in range(chans):
            # take the column header from the 3 comment line, but
            head = lines[2 + (3 + iters)*k][:-2].split('\t')[1].encode('utf-8')
            i0 = 3 + (3 + iters)*k
            i1 = i0 + iters
            data[head] = np.array([k[:-2].split('\t')[1] for k in lines[i0:i1:1]])
            data[head] = data[head].astype(np.float64)
        time = np.array([k[:-2].split('\t')[0] for k in lines[i0:i1:1]])
        df = pd.DataFrame(data, index=time.astype(np.float64))
        df.index.name = lines[0][:-2]
        return df


if __name__ == '__main__':

    pass
