# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:53:59 2014

@author: dave
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import dict
from io import open as opent
from builtins import range
from builtins import str
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import object

__author__ = 'David Verelst'
__license__ = 'GPL'
__version__ = '0.5'

import os
import copy
import struct
import math
from time import time
import codecs

import scipy
import scipy.integrate as integrate
import array
import numpy as np
import pandas as pd

# misc is part of prepost, which is available on the dtu wind gitlab server:
# https://gitlab.windenergy.dtu.dk/dave/prepost
from wetb.prepost import misc
# wind energy python toolbox, available on the dtu wind redmine server:
# http://vind-redmine.win.dtu.dk/projects/pythontoolbox/repository/show/fatigue_tools
from wetb.hawc2.Hawc2io import ReadHawc2
from wetb.fatigue_tools.fatigue import eq_load


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
                'direction'])

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
        if file_name[-4:] in ['.htc', '.sel', '.dat', '.log']:
            file_name = file_name[:-4]
        # FIXME: since HAWC2 will always have lower case output files, convert
        # any wrongly used upper case letters to lower case here
        self.file_name = file_name.lower()
        FileName = os.path.join(self.file_path, self.file_name)

        ReadOnly = 0 if readdata else 1
        super(LoadResults, self).__init__(FileName, ReadOnly=ReadOnly)
        ChVec = [] if usecols is None else usecols
        self.sig = super(LoadResults, self).__call__(ChVec=ChVec)

        if self.debug:
            stop = time() - start
            print('time to load HAWC2 file:', stop, 's')

    def reformat_sig_details(self):
        """Change HAWC2 output description of the channels short descriptive
        strings, usable in plots

        obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
        """

        # CONFIGURATION: mappings between HAWC2 and short good output:
        change_list = []
        change_list.append( ['original', 'new improved'] )

#        change_list.append( ['Mx coo: hub1','blade1 root bending: flap'] )
#        change_list.append( ['My coo: hub1','blade1 root bending: edge'] )
#        change_list.append( ['Mz coo: hub1','blade1 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub2','blade2 root bending: flap'] )
#        change_list.append( ['My coo: hub2','blade2 root bending: edge'] )
#        change_list.append( ['Mz coo: hub2','blade2 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub3','blade3 root bending: flap'] )
#        change_list.append( ['My coo: hub3','blade3 root bending: edge'] )
#        change_list.append( ['Mz coo: hub3','blade3 root bending: torsion'] )

        change_list.append(['Mx coo: blade1', 'blade1 flap'])
        change_list.append(['My coo: blade1', 'blade1 edge'])
        change_list.append(['Mz coo: blade1', 'blade1 torsion'])

        change_list.append(['Mx coo: blade2', 'blade2 flap'])
        change_list.append(['My coo: blade2', 'blade2 edge'])
        change_list.append(['Mz coo: blade2', 'blade2 torsion'])

        change_list.append(['Mx coo: blade3', 'blade3 flap'])
        change_list.append(['My coo: blade3', 'blade3 edeg'])
        change_list.append(['Mz coo: blade3', 'blade3 torsion'])

        change_list.append(['Mx coo: hub1', 'blade1 out-of-plane'])
        change_list.append(['My coo: hub1', 'blade1 in-plane'])
        change_list.append(['Mz coo: hub1', 'blade1 torsion'])

        change_list.append(['Mx coo: hub2', 'blade2 out-of-plane'])
        change_list.append(['My coo: hub2', 'blade2 in-plane'])
        change_list.append(['Mz coo: hub2', 'blade2 torsion'])

        change_list.append(['Mx coo: hub3', 'blade3 out-of-plane'])
        change_list.append(['My coo: hub3', 'blade3 in-plane'])
        change_list.append(['Mz coo: hub3', 'blade3 torsion'])
        # this one will create a false positive for tower node nr1
        change_list.append(['Mx coo: tower', 'tower top momemt FA'])
        change_list.append(['My coo: tower', 'tower top momemt SS'])
        change_list.append(['Mz coo: tower', 'yaw-moment'])

        change_list.append(['Mx coo: chasis', 'chasis momemt FA'])
        change_list.append(['My coo: chasis', 'yaw-moment chasis'])
        change_list.append(['Mz coo: chasis', 'chasis moment SS'])

        change_list.append(['DLL inp  2:  2', 'tower clearance'])

        self.ch_details_new = np.ndarray(shape=(self.Nch, 3), dtype='<U100')

        # approach: look for a specific description and change it.
        # This approach is slow, but will not fail if the channel numbers change
        # over different simulations
        for ch in range(self.Nch):
            # the change_list will always be slower, so this loop will be
            # inside the bigger loop of all channels
            self.ch_details_new[ch, :] = self.ch_details[ch, :]
            for k in range(len(change_list)):
                if change_list[k][0] == self.ch_details[ch, 0]:
                    self.ch_details_new[ch, 0] = change_list[k][1]
                    # channel description should be unique, so delete current
                    # entry and stop looking in the change list
                    del change_list[k]
                    break

#        self.ch_details_new = ch_details_new

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
        ch_aero = set(['Cl', 'Cd', 'Alfa', 'Vrel', 'Tors_e', 'Alfa'])
        ch_aerogrid = set(['a_grid', 'am_grid'])

        # also safe as df
#        cols = set(['bearing_name', 'sensortag', 'bodyname', 'chi',
#                    'component', 'pos', 'coord', 'sensortype', 'radius',
#                    'blade_nr', 'units', 'output_type', 'io_nr', 'io', 'dll',
#                    'azimuth', 'flap_nr'])
        df_dict = {col: [] for col in self.cols}
        df_dict['ch_name'] = []

        # scan through all channels and see which can be converted
        # to sensible unified name
        for ch in range(self.Nch):
            items = self.ch_details[ch, 2].split(' ')
            # remove empty values in the list
            items = misc.remove_items(items, '')

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
                coord = items[5]
                bodyname = items[1].replace('Mbdy:', '')
                # set nodenr to sortable way, include leading zeros
                # node numbers start with 0 at the root
                nodenr = '%03i' % int(items[3])
                # skip the attached the component
                # sensortype = items[0][:-2]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'momentvec'
                component = items[0][-1:len(items[0])]
                # the tag only exists if defined
                if len(items) > 6:
                    sensortag = ' '.join(items[6:])
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
            elif self.ch_details[ch, 2].startswith('Force'):
                coord = items[6]
                bodyname = items[2].replace('Mbdy:', '')
                nodenr = '%03i' % int(items[4])
                # skipe the attached the component
                # sensortype = items[0]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'forcevec'
                component = items[1][1]
                if len(items) > 7:
                    sensortag = ' '.join(items[7:])
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
                coord = items[8]
                bodyname = items[3].replace('Mbdy:', '')
                # element numbers start with 1 at the root
                elementnr = '%03i' % int(items[5])
                zrel = '%04.2f' % float(items[6].replace('Z-rel:', ''))
                # skip the attached the component
                #sensortype = ''.join(items[0:2])
                # or give the sensor type the same name as in HAWC2
                tmp = self.ch_details[ch, 0].split(' ')
                sensortype = tmp[0]
                if sensortype.startswith('State'):
                    sensortype += ' ' + tmp[1]
                component = items[2]
                if len(items) > 8:
                    sensortag = ' '.join(items[9:])
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
                if items[1][0] == ':echo':
                    # hawc_dll named case (case 3) is polluted with colons
                    items = self.ch_details[ch,2].replace(':', '')
                    items = items.split(' ')
                    items = misc.remove_items(items, '')
                    dll = items[1]
                    io = items[2]
                    io_nr = items[3]
                    tag = 'DLL-%s-%s-%s' % (dll, io, io_nr)
                    sensortag = ''
                # case 2: no reference to dll name
                elif self.ch_details[ch,2].startswith('DLL'):
                    dll = items[2]
                    io = items[3]
                    io_nr = items[5]
                    sensortag = ' '.join(items[6:])
                    # and tag it
                    tag = 'DLL-%s-%s-%s' % (dll,io,io_nr)
                # case 1: type2 dll name is given
                else:
                    dll = items[0]
                    io = items[1]
                    io_nr = items[2]
                    sensortag = ' '.join(items[3:])
                    tag = 'DLL-%s-%s-%s' % (dll, io, io_nr)

                # save all info in the dict
                channelinfo = {}
                channelinfo['dll'] = dll
                channelinfo['io'] = io
                channelinfo['io_nr'] = io_nr
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch, 1]

            # -----------------------------------------------------------------
            # BEARING OUTPUS
            # bea1 angle_speed       rpm      shaft_nacelle angle speed
            elif self.ch_details[ch, 0].startswith('bea'):
                output_type = self.ch_details[ch, 0].split(' ')[1]
                bearing_name = items[0]
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
            # AERO CL, CD, CM, VREL, ALFA, LIFT, DRAG, etc
            # Cl, R=  0.5     deg      Cl of blade  1 at radius   0.49
            # Azi  1          deg      Azimuth of blade  1
            elif self.ch_details[ch, 0].split(',')[0] in ch_aero:
                dscr_list = self.ch_details[ch, 2].split(' ')
                dscr_list = misc.remove_items(dscr_list, '')

                sensortype = self.ch_details[ch, 0].split(',')[0]
                radius = dscr_list[-1]
                # is this always valid?
                blade_nr = self.ch_details[ch, 2].split('blade  ')[1][0]
                # sometimes the units for aero sensors are wrong!
                units = self.ch_details[ch, 1]
                # there is no label option

                # and tag it
                tag = '%s-%s-%s' % (sensortype, blade_nr, radius)
                # save all info in the dict
                channelinfo = {}
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = float(radius)
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # for the induction grid over the rotor
            # a_grid, azi    0.00 r   1.74
            elif self.ch_details[ch, 0].split(',')[0] in ch_aerogrid:
                items = self.ch_details[ch, 0].split(',')
                sensortype = items[0]
                items2 = items[1].split(' ')
                items2 = misc.remove_items(items2, '')
                azi = items2[1]
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
# Induc. Vx, locco, R=  1.4 // Induced wsp Vx of blade  1 at radius   1.37, local ae coo.
# Induc. Vy, blco, R=  1.4 // Induced wsp Vy of blade  1 at radius   1.37, local bl coo.
# Induc. Vz, glco, R=  1.4 // Induced wsp Vz of blade  1 at radius   1.37, global coo.
# Induc. Vx, rpco, R=  8.4 // Induced wsp Vx of blade  1 at radius   8.43, RP. coo.
            elif self.ch_details[ch, 0].strip()[:5] == 'Induc':
                items = self.ch_details[ch, 2].split(' ')
                items = misc.remove_items(items, '')
                blade_nr = int(items[5])
                radius = float(items[8].replace(',', ''))
                items = self.ch_details[ch, 0].split(',')
                coord = items[1].strip()
                component = items[0][-2:]
                units = self.ch_details[ch, 1]
                # and tag it
                rpl = (coord, blade_nr, component, radius)
                tag = 'induc-%s-blade-%1i-%s-r-%03.02f' % rpl
                # save all info in the dict
                channelinfo = {}
                channelinfo['blade_nr'] = blade_nr
                channelinfo['sensortype'] = 'induction'
                channelinfo['radius'] = radius
                channelinfo['coord'] = coord
                channelinfo['component'] = component
                channelinfo['units'] = units
                channelinfo['chi'] = ch

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
                x = items[-2][:-1]
                y = items[-1]

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
            # WSP gl. coo.,Vx
            elif self.ch_details[ch, 0].startswith('WSP gl.'):
                units = self.ch_details[ch, 1]
                direction = self.ch_details[ch, 0].split(',')[1]
                tmp = self.ch_details[ch, 2].split('pos')[1]
                x, y, z = tmp.split(',')
                x, y, z = x.strip(), y.strip(), z.strip()

                # and tag it
                tag = 'windspeed-global-%s-%s-%s-%s' % (direction, x, y, z)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = 'global'
                channelinfo['pos'] = (x, y, z)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # WIND SPEED AT BLADE
            # 0: WSP Vx, glco, R= 61.5
            # 2: Wind speed Vx of blade  1 at radius  61.52, global coo.
            elif self.ch_details[ch, 0].startswith('WSP V'):
                units = self.ch_details[ch, 1].strip()
                direction = self.ch_details[ch, 0].split(' ')[1].strip()
                blade_nr = self.ch_details[ch, 2].split('blade')[1].strip()[:2]
                radius = self.ch_details[ch, 2].split('radius')[1].split(',')[0]
                coord = self.ch_details[ch, 2].split(',')[1].strip()

                radius = radius.strip()
                blade_nr = blade_nr.strip()

                # and tag it
                rpl = (direction, blade_nr, radius, coord)
                tag = 'wsp-blade-%s-%s-%s-%s' % rpl
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['direction'] = direction
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['radius'] = float(radius)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # FLAP ANGLE
            # 2: Flap angle for blade  3 flap number  1
            elif self.ch_details[ch, 0][:7] == 'setbeta':
                units = self.ch_details[ch, 1].strip()
                blade_nr = self.ch_details[ch, 2].split('blade')[1].strip()
                blade_nr = blade_nr.split(' ')[0].strip()
                flap_nr = self.ch_details[ch, 2].split(' ')[-1].strip()

                radius = radius.strip()
                blade_nr = blade_nr.strip()

                # and tag it
                tag = 'setbeta-bladenr-%s-flapnr-%s' % (blade_nr, flap_nr)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['flap_nr'] = int(flap_nr)
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # ignore all the other cases we don't know how to deal with
            else:
                # if we get here, we don't have support yet for that sensor
                # and hence we can't save it. Continue with next channel
                continue

            # -----------------------------------------------------------------
            # ignore if we have a non unique tag
            if tag in self.ch_dict:
                jj = 1
                while True:
                    tag_new = tag + '_v%i' % jj
                    if tag_new in self.ch_dict:
                        jj += 1
                    else:
                        tag = tag_new
                        break
#                msg = 'non unique tag for HAWC2 results, ignoring: %s' % tag
#                logging.warn(msg)
#            else:
            self.ch_dict[tag] = copy.copy(channelinfo)

            # -----------------------------------------------------------------
            # save in for DataFrame format
            cols_ch = set(channelinfo.keys())
            for col in cols_ch:
                df_dict[col].append(channelinfo[col])
            # the remainder columns we have not had yet. Fill in blank
            for col in (self.cols - cols_ch):
                df_dict[col].append('')
            df_dict['ch_name'].append(tag)

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
        df_dict['ch_name'] = []
        for ch_name, channelinfo in self.ch_dict.items():
            cols_ch = set(channelinfo.keys())
            for col in cols_ch:
                df_dict[col].append(channelinfo[col])
            # the remainder columns we have not had yet. Fill in blank
            for col in (cols - cols_ch):
                df_dict[col].append('')
            df_dict['ch_name'].append(ch_name)

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
    command.
    """
    # because the formatting is really weird, we need to sanatize it a bit
    with opent(fname, 'r') as f:
        # read the header from line 3
        f.readline()
        f.readline()
        header = f.readline().replace('\r', '').replace('\n', '')
        cols = [k.strip().replace(' ', '_') for k in header.split('#')[1:]]

#    data = pd.read_fwf(fname, skiprows=3, header=None)
#    pd.read_table(fname, sep='  ', skiprows=3)
#    data.index.names = cols

    data = np.loadtxt(fname, skiprows=3)
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
    FILE = opent(fname)
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


def ReadEigenStructure(file_path, file_name, debug=False, max_modes=500):
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

    FILE = opent(os.path.join(file_path, file_name))
    lines = FILE.readlines()
    FILE.close()

    header_lines = 8

    # we now the number of modes by having the number of lines
    nrofmodes = len(lines) - header_lines

    modes_arr = np.ndarray((3, nrofmodes))

    for i, line in enumerate(lines):
        if i > max_modes:
            # cut off the unused rest
            modes_arr = modes_arr[:, :i]
            break

        # ignore the header
        if i < header_lines:
            continue

        # split up mode nr from the rest
        parts = line.split(':')
        # modenr = int(parts[1])
        # get fd, fn and damping, but remove all empty items on the list
        modes_arr[:, i-header_lines]=misc.remove_items(parts[2].split(' '), '')

    return modes_arr


class UserWind(object):
    """
    """

    def __init__(self):
        pass

    def __call__(self, z_h, r_blade_tip, a_phi=None, shear_exp=None, nr_hor=3,
                 nr_vert=20, h_ME=500.0, fname=None, wdir=None):
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

        fname : str, default=None
            When specified, the HAWC2 user defined veer input file will be
            written.

        wdir : float, default=None
            A constant veer angle, or yaw angle. Equivalent to setting the
            wind direction. Angle in degrees.

        Returns
        -------

        None

        """

        x, z = self.create_coords(z_h, r_blade_tip, nr_vert=nr_vert,
                                  nr_hor=nr_hor)
        if a_phi is not None:
            phi_rad = self.veer_ekman_mod(z, z_h, h_ME=h_ME, a_phi=a_phi)
            assert len(phi_rad) == nr_vert
        else:
            nr_vert = len(z)
            phi_rad = np.zeros((nr_vert,))
        # add any yaw error on top of
        if wdir is not None:
            # because wdir cw positive, and phi veer ccw positive
            phi_rad -= (wdir*np.pi/180.0)
        u, v, w, xx, zz = self.decompose_veer(phi_rad, x, z)
        # scale the shear on top of that
        if shear_exp is not None:
            shear = self.shear_powerlaw(zz, z_h, shear_exp)
            uu = u*shear[:,np.newaxis]
            vv = v*shear[:,np.newaxis]
            ww = w*shear[:,np.newaxis]
        # and write to a file
        if fname is not None:
            self.write_user_defined_shear(fname, uu, vv, ww, xx, zz)

    def create_coords(self, z_h, r_blade_tip, nr_vert=3, nr_hor=20):
        """
        Utility to create the coordinates of the wind field based on hub heigth
        and blade length.
        """
        # take 15% extra space after the blade tip
        z = np.linspace(0, z_h + r_blade_tip*1.15, nr_vert)
        # along the horizontal, coordinates with 0 at the rotor center
        x = np.linspace(-r_blade_tip*1.15, r_blade_tip*1.15, nr_hor)

        return x, z

    def shear_powerlaw(self, z, z_ref, a):
        profile = np.power(z/z_ref, a)
        # when a negative, make sure we return zero and not inf
        profile[np.isinf(profile)] = 0.0
        return profile

    def veer_ekman_mod(self, z, z_h, h_ME=500.0, a_phi=0.5):
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

        :math:`a_{\\varphi} \\approx 0.5` parameter for the modified
            Ekman veer distribution. Values vary between -1.2 and 0.5.

        returns
        -------

        phi_rad : ndarray
            veer angle in radians

        """

        t1 = np.exp(-math.sqrt(z_h / h_ME))
        t2 = (z - z_h) / math.sqrt(z_h * h_ME)
        t3 = (1.0 - (z-z_h)/(2.0*math.sqrt(z_h*h_ME)) - (z-z_h)/(4.0*z_h))

        return a_phi * t1 * t2 * t3

    def decompose_veer(self, phi_rad, x, z):
        """
        Convert a veer angle into u, v, and w components, ready for the
        HAWC2 user defined veer input file.

        Paramters
        ---------

        phi_rad : ndarray
            veer angle in radians

        method : str, default=linear
            'linear' for a linear veer, 'ekman_mod' for modified ekman method

        Returns
        -------

        u, v, w, v_coord, w_coord

        """

        nr_hor = len(x)
        nr_vert = len(z)
        assert len(phi_rad) == nr_vert

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

        return u_full, v_full, w_full, x, z

    def load_user_defined_veer(self, fname):
        """
        Load a user defined veer and shear file as used for HAWC2

        Returns
        -------

        u_comp, v_comp, w_comp, v_coord, w_coord, phi_deg
        """
        blok = 0
        bloks = {}
        with opent(fname) as f:
            for i, line in enumerate(f.readlines()):
                if line.strip()[0] == '#' and blok > 0:
                    bloks[blok] = i
                    blok += 1
                elif line.strip()[0] == '#':
                    continue
                elif blok == 0:
                    items = line.split(' ')
                    items = misc.remove_items(items, '')
                    nr_hor, nr_vert = int(items[0]), int(items[1])
                    blok += 1
#        nr_lines = i

        k = nr_hor + 4*nr_vert + 7
        v_comp = np.genfromtxt(fname, skiprows=3, skip_footer=i-3-3-nr_vert)
        u_comp = np.genfromtxt(fname, skiprows=3+1+nr_vert,
                               skip_footer=i-3-3-nr_vert*2)
        w_comp = np.genfromtxt(fname, skiprows=3+2+nr_vert*2,
                               skip_footer=i-3-3-nr_vert*3)
        v_coord = np.genfromtxt(fname, skiprows=3+3+nr_vert*3,
                                skip_footer=i-3-3-nr_vert*3-3)
        w_coord = np.genfromtxt(fname, skiprows=3+3+nr_vert*3+4,
                                skip_footer=i-k)
        phi_deg = np.arctan(v_comp[:, 0]/u_comp[:, 0])*180.0/np.pi

        return u_comp, v_comp, w_comp, v_coord, w_coord, phi_deg

    def write_user_defined_shear(self, fname, u, v, w, v_coord, w_coord,
                                 fmt_uvw='% 08.05f', fmt_coord='% 8.02f'):
        """
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

        # and create the input file
        with open(fname, 'wb') as fid:
            fid.write(b'# User defined shear file\n')
            fid.write(b'%i %i # nr_hor (v), nr_vert (w)\n' % (nr_hor, nr_vert))
            h1 = b'normalized with U_mean, nr_hor (v) rows, nr_vert (w) columns'
            fid.write(b'# v component, %s\n' % h1)
            np.savetxt(fid, v, fmt=fmt_uvw, delimiter='  ')
            fid.write(b'# u component, %s\n' % h1)
            np.savetxt(fid, u, fmt=fmt_uvw, delimiter='  ')
            fid.write(b'# w component, %s\n' % h1)
            np.savetxt(fid, w, fmt=fmt_uvw, delimiter='  ')
            h2 = b'# v coordinates (along the horizontal, nr_hor, 0 rotor center)'
            fid.write(b'%s\n' % h2)
            np.savetxt(fid, v_coord.reshape((v_coord.size, 1)), fmt=fmt_coord)
            h3 = b'# w coordinates (zero is at ground level, height, nr_hor)'
            fid.write(b'%s\n' % h3)
            np.savetxt(fid, w_coord.reshape((w_coord.size, 1)), fmt=fmt_coord)


class WindProfiles(object):

    def __init__(self):
        pass

    def powerlaw(self, z, z_ref, a):
        profile = np.power(z/z_ref, a)
        # when a negative, make sure we return zero and not inf
        profile[np.isinf(profile)] = 0.0
        return profile

    def veer_ekman_mod(self, z, z_h, h_ME=500.0, a_phi=0.5):
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

        :math:`a_{\\varphi} \\approx 0.5` parameter for the modified
            Ekman veer distribution. Values vary between -1.2 and 0.5.

        returns
        -------

        phi_rad : ndarray
            veer angle in radians as function of height

        """

        t1 = np.exp(-math.sqrt(z_h / h_ME))
        t2 = (z - z_h) / math.sqrt(z_h * h_ME)
        t3 = ( 1.0 - (z-z_h)/(2.0*math.sqrt(z_h*h_ME)) - (z-z_h)/(4.0*z_h) )

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

        with codecs.opent(fname, 'r', enc) as f:
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
