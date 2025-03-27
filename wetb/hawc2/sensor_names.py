#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:11:48 2025

@author: dave
"""

import re
import copy
import pandas as pd

from wetb.prepost import misc


def unified_channel_names(ChInfo):
    """Create consistant and unique channel names for a HAWC2 result file.

    Parameters
    ----------
    ChInfo : List of list
        The list of list as given by wetb.hawc2.ReadHawc2.ChInfo.

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

    Returns
    -------
    None.

    """

    # save them in a dictionary, use the new coherent naming structure
    # as the key, and as value again a dict that hols all the different
    # classifications: (chi, channel nr), (coord, coord), ...
    ch_dict = dict()

    # all columns for the output
    cols = ['bearing_name', 'sensortag', 'bodyname', 'chi', 'component',
            'pos', 'coord', 'sensortype', 'radius', 'blade_nr', 'units',
            'output_type', 'io_nr', 'io', 'dll', 'azimuth', 'flap_nr',
            'direction', 'wake_source_nr', 'center', 's', 'srel',
            'radius_actual']
    # so we can look up missing columns later
    # don't use the set for building df_dict since the order will become random
    colsset = set(cols)

    # some channel ID's are unique, use them
    ch_unique = set(['Omega', 'Ae rot. torque', 'Ae rot. power',
                     'Ae rot. thrust', 'Time', 'Azi  1'])
    ch_aero = set(['Cl', 'Cd', 'Cm', 'Alfa', 'Vrel', 'Tors_e', 'Alfa',
                   'Lift', 'Drag'])
    ch_aerogrid = set(['a_grid', 'am_grid', 'CT', 'CQ'])

    # also safe as df
    df_dict = {col: [] for col in cols}
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
    for ich in range(len(ChInfo[0])):

        ch_id = ChInfo[0][ich]
        ch_unit = ChInfo[1][ich]
        ch_desc = ChInfo[2][ich]

        # if len(ch_id) < 1 or len(ch_desc) < 1:
        #     continue

        items_ch0 = ch_id.split()
        items_ch2 = ch_desc.split()

        dll = False

        # be carefull, identify only on the starting characters, because
        # the signal tag can hold random text that in some cases might
        # trigger a false positive

        # -----------------------------------------------------------------
        # check for all the unique channel descriptions
        if ch_id.strip() in ch_unique:
            tag = ch_id.strip()
            channelinfo = {}
            channelinfo['units'] = ch_unit
            channelinfo['sensortag'] = ch_desc
            channelinfo['chi'] = ich

        # -----------------------------------------------------------------
        # or in the long description:
        #    0          1        2      3  4    5     6 and up
        # MomentMz Mbdy:blade nodenr:   5 coo: blade  TAG TEXT
        elif ch_desc.startswith('MomentM'):
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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit

        # -----------------------------------------------------------------
        #   0    1      2        3       4  5     6     7 and up
        # Force  Fx Mbdy:blade nodenr:   2 coo: blade  TAG TEXT
        elif ch_desc.startswith('Force  F'):
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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit

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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit

        # -----------------------------------------------------------------
        # ELEMENT STATES: pos, vel, acc, rot, ang
        #   0    1  2      3       4      5   6         7    8
        # State pos x  Mbdy:blade E-nr:   1 Z-rel:0.00 coo: blade
        #   0           1     2    3        4    5   6         7     8     9+
        # State_rot proj_ang tx Mbdy:bname E-nr: 1 Z-rel:0.00 coo: cname  label
        # State_rot omegadot tz Mbdy:bname E-nr: 1 Z-rel:1.00 coo: cname  label
        elif ch_desc.startswith('State'):
#                 or ch_details[ich,0].startswith('euler') \
#                 or ch_details[ich,0].startswith('ax') \
#                 or ch_details[ich,0].startswith('omega') \
#                 or ch_details[ich,0].startswith('proj'):
            coord = items_ch2[8]
            bodyname = items_ch2[3].replace('Mbdy:', '')
            # element numbers start with 1 at the root
            elementnr = '%03i' % int(items_ch2[5])
            zrel = '%04.2f' % float(items_ch2[6].replace('Z-rel:', ''))
            # skip the attached the component
            #sensortype = ''.join(items[0:2])
            # or give the sensor type the same name as in HAWC2
            tmp = ch_id.split(' ')
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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit

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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit

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
        elif ch_id.startswith('DLL'):
            # case 3
            if items_ch2[0] == 'hawc_dll':
                # hawc_dll named case (case 3) is polluted with colons
                dll = items_ch2[1].replace(':', '')
                io = items_ch2[2]
                io_nr = items_ch2[4]
                tag = 'DLL-%s-%s-%s' % (dll, io, io_nr)
                sensortag = ''
            # case 2: no reference to dll name
            elif ch_desc.startswith('DLL'):
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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            channelinfo['units'] = ch_unit
            channelinfo['sensortype'] = 'dll-io'

        # -----------------------------------------------------------------
        # BEARING OUTPUS
        # bea1 angle_speed       rpm      shaft_nacelle angle speed
        elif ch_id.startswith('bea'):
            output_type = ch_id.split(' ')[1]
            bearing_name = items_ch2[0]
            units = ch_unit
            # there is no label option for the bearing output

            # and tag it
            tag = 'bearing-%s-%s-%s' % (bearing_name, output_type, units)
            # save all info in the dict
            channelinfo = {}
            channelinfo['bearing_name'] = bearing_name
            channelinfo['output_type'] = output_type
            channelinfo['units'] = units
            channelinfo['chi'] = ich

        # -----------------------------------------------------------------
        # AS DEFINED IN: ch_aero
        # AERO CL, CD, CM, VREL, ALFA, LIFT, DRAG, etc
        # Cl, R=  0.5     deg      Cl of blade  1 at radius   0.49
        # Azi  1          deg      Azimuth of blade  1
        #
        # ch_desc:
        # Angle of attack of blade   1 at radius   8.59 FOLLOWD BY USER LABEL
        #
        # NOTE THAT RADIUS FROM ch_id REFERS TO THE RADIUS
        # YOU ASKED FOR, AND ch_desc IS WHAT YOU GET, which is
        # still based on a mean radius (deflections change the game)
        elif ch_id.split(',')[0] in ch_aero:
            sensortype = ch_id.split(',')[0]

            # sometimes the units for aero sensors are wrong!
            units = ch_unit
            # there is no label option

            # Blade number is identified as the first integer in the string
            # blade_nr = re.search(r'\d+', ch_desc).group()
            # blade_nr = int(blade_nr)

            # actual radius
            rq = r'\.*of blade\s*(\d) at radius\s*([-+]?\d*\.\d+|\d+)'
            s = ch_desc
            blade_nr, radius_actual = re.findall(rq, s)[0]
            blade_nr = int(blade_nr)

            # radius what you asked for, identified as the last float in the string
            s = ch_id
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
            channelinfo['chi'] = ich

        # -----------------------------------------------------------------
        # for the induction grid over the rotor
        # a_grid, azi    0.00 r   1.74
        elif ch_id.split(',')[0] in ch_aerogrid:
            items_ = ch_id.split(',')
            sensortype = items_[0]
            items2 = items_[1].split(' ')
            items2 = misc.remove_items(items2, '')
            azi = items2[1]
            # radius what you asked for
            radius = items2[3]
            units = ch_unit
            # and tag it
            tag = '%s-azi-%s-r-%s' % (sensortype,azi,radius)
            # save all info in the dict
            channelinfo = {}
            channelinfo['sensortype'] = sensortype
            channelinfo['radius'] = float(radius)
            channelinfo['azimuth'] = float(azi)
            channelinfo['units'] = units
            channelinfo['chi'] = ich

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
        elif ch_id.strip()[:5] == 'Induc':

            coord = ch_desc.split(', ')[1].strip()
            blade_nr = int(items_ch2[5])

            # radius what you get
            #  radius = float(items[8].replace(',', ''))
            # radius what you asked for, identified as the last float in the string
#                s = ch_desc
#                radius = float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])
            radius = float(items_ch2[8][:-1])

            component = items_ch2[2]
            units = ch_unit

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
            channelinfo['chi'] = ich

        # -----------------------------------------------------------------
        # MORE AERO SENSORS
        # Ae intfrc Fx, rpco, R=  0.0
        #     Aero int. force Fx of blade  1 at radius   0.00, RP coo.
        # Ae secfrc Fy, R= 25.0
        #     Aero force  Fy of blade  1 at radius  24.11
        # Ae pos x, glco, R= 88.2
        #     Aero position x of blade  1 at radius  88.17, global coo.
        elif ch_id.strip()[:2] == 'Ae':
            units = ch_unit
            # Blade number is identified as the first integer in the string
            blade_nr = re.search(r'\d+', ch_desc).group()
            blade_nr = int(blade_nr)
            # radius what you get
            tmp = ch_desc.split('radius ')[1].strip()
            tmp = tmp.split(',')
            # radius = float(tmp[0])
            # radius what you asked for, identified as the last float in the string
            s = ch_desc
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
            channelinfo['chi'] = ich

            rpl = (coord, blade_nr, sensortype, component, radius)
            tag = 'aero-%s-blade-%1i-%s-%s-r-%03.01f' % rpl

        # TODO: wind speed
        # some spaces have been trimmed here
        # WSP gl. coo.,Vy          m/s
        # // Free wind speed Vy, gl. coo, of gl. pos   0.00,  0.00,  -2.31
        # WSP gl. coo.,Vdir_hor          deg
        # Free wind speed Vdir_hor, gl. coo, of gl. pos  0.00,  0.00, -2.31

        # -----------------------------------------------------------------
        # Water surface gl. coo, at gl. coo, x,y=   0.00,   0.00
        # Water vel., x-dir, at gl. coo, x,y,z=   0.00,   0.00,   1.00
        elif ch_desc.startswith('Water'):
            units = ch_unit
            channelinfo = {}

            # surface, vel or acc?
            if items_ch2[1]=='surface':
                # but remove the comma
                x = items_ch2[-2][:-1]
                y = items_ch2[-1]
                # and tag it
                tag = 'watersurface-global-%s-%s' % (x, y)
                channelinfo['pos'] = (float(x), float(y))
            else:
                # but remove the comma
                x = items_ch2[-3][:-1]
                y = items_ch2[-2][:-1]
                z = items_ch2[-1]
                tag = f'{items_ch0[0]}-{x}-{y}-{z}'

            # save all info in the dict
            channelinfo['coord'] = 'global'
            channelinfo['units'] = units
            channelinfo['chi'] = ich

        # -----------------------------------------------------------------
        # WIND SPEED
        elif ch_desc.startswith('Free wind speed'):
            units = ch_unit
            direction = ch_id.split(',')[1]
            # WSP gl. coo.,Vx
            # Free wind speed Vx, gl. coo, of gl. pos    0.00,   0.00,  -6.00  LABEL
            if ch_desc.startswith('Free '):
                tmp = ch_desc.split('pos')[1]
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
            channelinfo['chi'] = ich
            channelinfo['sensortag'] = sensortag
            # FIXME: direction is the same as component, right?
            channelinfo['direction'] = direction
            channelinfo['sensortype'] = 'wsp-global'

        # WIND SPEED AT BLADE
        # 0: WSP Vx, glco, R= 61.5
        # 2: Wind speed Vx of blade  1 at radius  61.52, global coo.
        elif ch_id.startswith('WSP V'):
            units = ch_unit.strip()
            tmp = ch_id.split(' ')[1].strip()
            direction = tmp.replace(',', '')
            coord = ch_desc.split(',')[1].split()[0]
            # Blade number is identified as the first integer in the string
            blade_nr = re.search(r'\d+', ch_desc).group()
            blade_nr = int(blade_nr)

            # radius what you get
            # radius = ch_desc.split('radius')[1].split(',')[0]
            # radius = radius.strip()
            # radius what you asked for, identified as the last float in the string
            s=ch_desc
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
            channelinfo['chi'] = ich
            channelinfo['sensortype'] = 'wsp-blade'

        # FLAP ANGLE
        # 2: Flap angle for blade  3 flap number  1
        elif ch_id[:7] == 'setbeta':
            units = ch_unit.strip()
            # Blade number is identified as the first integer in the string
            blade_nr = re.search(r'\d+', ch_desc).group()
            blade_nr = int(blade_nr)
            flap_nr = ch_desc.split(' ')[-1].strip()

            # and tag it
            tag = 'setbeta-bladenr-%s-flapnr-%s' % (blade_nr, flap_nr)
            # save all info in the dict
            channelinfo = {}
            channelinfo['flap_nr'] = int(flap_nr)
            channelinfo['blade_nr'] = blade_nr
            channelinfo['units'] = units
            channelinfo['chi'] = ich

        # harmonic channel output
        # Harmonic
        # Harmonic sinus function
        elif ch_id[:7] == 'Harmoni':

            func_name = ' '.join(ch_unit.split(' ')[1:])

            channelinfo = {}
            channelinfo['output_type'] = func_name
            channelinfo['sensortype'] = 'harmonic'
            channelinfo['chi'] = ich

            base = ch_desc.strip().lower().replace(' ', '_')
            tag = base + '_0'
            if tag in ch_dict:
                tag_nr = int(tag.split('_')[-1]) + 1
                tag = base + '_%i' % tag_nr

        elif ch_id[:6] == 'a_norm':
            channelinfo = {}
            channelinfo['chi'] = ich
            channelinfo['units'] = ch_unit.strip()
            channelinfo['sensortype'] = 'aero'
            tag = 'aero-induc_a_norm'

        # wake   1 gl. pos pos_z  //  Wake pos_z of source   1, Met. coo.
        elif ch_id[:4] == 'wake':
            wake_nr = re.search(r'\d+', ch_id).group()
            comp = re.search(r'pos_([xyz])', ch_id).group(1)

            channelinfo = {}
            channelinfo['output_type'] = 'wake_pos'
            channelinfo['sensortype'] = 'wind_wake'
            channelinfo['component'] = comp
            channelinfo['units'] = ch_unit.strip()
            channelinfo['chi'] = ich
            channelinfo['wake_source_nr'] = int(wake_nr)
            channelinfo['coord'] = 'met'

            tag = 'wind_wake-wake_pos_%s_%s' % (comp, wake_nr)

        # ESYS line1 SENSOR            1
        elif ch_desc[:4] == 'ESYS':
            # body = re.findall(regex, ch_desc)
            body, outnr = re_esys.match(ch_desc).groups()

            channelinfo = {}
            channelinfo['output_type'] = 'esys'
            channelinfo['sensortype'] = 'esys'
            channelinfo['io_nr'] = int(outnr)
            channelinfo['units'] = ch_unit.strip()
            channelinfo['chi'] = ich

            tag = 'esys-%s-%s' % (body, outnr)

        elif ch_desc[:4] == 'FORC':
            # body = re.findall(regex, ch_desc)
            dllname, outnr = re_force.match(ch_desc).groups()

            channelinfo = {}
            channelinfo['output_type'] = 'force-dll'
            channelinfo['sensortype'] = 'force-dll'
            channelinfo['io_nr'] = int(outnr)
            channelinfo['units'] = ch_unit.strip()
            channelinfo['chi'] = ich

            tag = 'force-%s-%s' % (dllname, outnr)

        # -----------------------------------------------------------------
        # If all this fails, just combine channel id and its tag if present
        else:
            tag = f'{ch_id}'
            channelinfo = {}
            channelinfo['chi'] = ich
            channelinfo['units'] = ch_unit.strip()

        # -----------------------------------------------------------------
        # add a v_XXX tag in case the channel already exists
        if tag in ch_dict:
            jj = 1
            while True:
                tag_new = tag + '_v%i' % jj
                if tag_new in ch_dict:
                    jj += 1
                else:
                    tag = tag_new
                    break

        ch_dict[tag] = copy.copy(channelinfo)

        # -----------------------------------------------------------------
        # save in for DataFrame format
        cols_ch = set(channelinfo.keys())
        for col in cols_ch:
            df_dict[col].append(channelinfo[col])
        # the remainder columns we have not had yet. Fill in blank
        for col in (colsset - cols_ch):
            df_dict[col].append('')
        df_dict['unique_ch_name'].append(tag)

    ch_df = pd.DataFrame(df_dict)
    ch_df.set_index('chi', inplace=True)

    return ch_dict, ch_df


def htc_channel_names(ChInfo):
    """Incomplete prototype to give every channel a unique channel name which
    is (nearly) identical to the channel names as defined in the htc output
    section. Instead of spaces, use colon (;) to seperate the different commands.

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

    for ch in range(len(ChInfo[0])):

        name = ChInfo[0][ch]
        name_items = misc.remove_items(name.split(' '), '')

        description = ChInfo[2][ch]
        descr_items = misc.remove_items(description.split(' '), '')

        unit = ChInfo[1][ch]

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

