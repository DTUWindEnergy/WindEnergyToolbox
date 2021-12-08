#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:20:28 2021

@author: dave
"""
import os
from os.path import join as pjoin
import warnings

import numpy as np
import pandas as pd

# from wetb.prepost import misc
from wetb.hawc2 import (HTCFile, AEFile, PCFile, StFile)

from wetb.bladed.readprj import ReadBladedProject, _DTYPE



# TODO: move to misc?
def get_circular_inertia(d1, t, rho):
    """ d1:     outer diameter in meters
        t:      thickness in meters
        rho:    density
    """
    d2 = d1 - 2*t
    A = np.pi / 4 * (d1**2 - d2**2)
    m = rho * A
    I_g = np.pi / 64 * (d1**4 - d2**4)
    I_m = I_g / A * m
    return m, A, I_g, I_m


# TODO: move to HTCFile?
def add_c2_arr_htc(htc, c2_arr, body):
    """add a c2_def in the form of an array to a htc c2_def section. Needs to
    be an existing body, and the current c2_def will be overwritten.
    """
    nr_nodes = c2_arr.shape[0]
    c2_def = htc['new_htc_structure'].get_subsection_by_name(body)['c2_def']

    keys = ['sec'] + [f'sec__{i}' for i in range(2, nr_nodes+1)]

    c2_def['nsec'].values[0] = nr_nodes
    for i, key in enumerate(keys):
        # control the formatting, otherwise it looks very messy in the htc file
        # this we do by converting the values to strings
        vals_fmt = [f'{k: 15.07f}' for k in c2_arr[i,:].tolist()]
        # node nr is stripped from leading spaces, so it doesn't help adding them
        values = [i+1] + vals_fmt
        if key in c2_def:
            c2_def[key].values[:] = values[:]
        else:
            c2_def.add_line(name=key, values=values)

    return htc


# TODO: move to HTCFile?
def get_c2def(htc, body_name):
    """
    """

    # select the c2def section from the shaft
    c2_def = htc['new_htc_structure'].get_subsection_by_name(body_name)['c2_def']
    # safe all c2_def positions in one array
    c2_arr = np.zeros((len(c2_def.keys())-1,4))
    # iterate over all nsec lines, but ignore the line that gives the number of points
    for i, sec in enumerate(c2_def.keys()[1:]):
        c2_arr[i,:] = c2_def[sec].values[1:]

    return pd.DataFrame(c2_arr, columns=['x', 'y', 'z', 'twist'])


class Convert2Hawc(ReadBladedProject):
    """Based on the BLADED user manual v4.8
    """

    def __init__(self, fname):
        super().__init__(fname)

    def get_pc(self, setname_filter=None):
        # ---------------------------------------------------------------------
        # we ignore the airfoils blocks at the start of the ADAT sub-section
        # ---------------------------------------------------------------------
        # nr_airfoils = prj.get_key('ADAT', 'NFOILS')[0,0]
        # airfoil_keys = ['NFOIL', 'FTYPE', 'NTHICK', 'NREYN']
        # airfoils = {}
        # for k in range(nr_airfoils):
        #     key_app = ''
        #     if k > 0:
        #         key_app = f'__{k+1}'
        #     airfoils[f'NFOIL_{k+1}'] = {}
        #     for key in airfoil_keys:
        # ---------------------------------------------------------------------

        # total number of pc sets
        nsets = self.get_key('ADAT', 'NSETS')[0,0]
        key_app = [''] + [f'__{k}' for k in range(2, nsets+1)]

        # FIXME: are all sets always named consistantly? Meaning only the
        # thickness changes in the setname, and the rest remains the same?
        # as a work-around: ignore sets that contains setname_filter in SETNAME
        if setname_filter is not None:
            drop = []
            for i, ka in enumerate(key_app):
                setname = self.get_key('ADAT', f'SETNAME{ka}')[0,0]
                if setname.find(setname_filter) > -1:
                    drop.append(ka)
            key_app = set(key_app) - set(drop)
            # update nsets
            nsets = len(key_app)

        # initiate PCFile objects
        pc = PCFile()
        tc_arr = np.ndarray((nsets), dtype=_DTYPE)
        pc_lst = []

        for i, ka in enumerate(key_app):

            # and assure cm is around quarter chord point
            # assert self.get_key('ADAT', 'XA')[0,0]==25
            if self.get_key('ADAT', 'XA')[0, 0] != 25:
                warnings.warn('Cm not calculated at 1/4 chord!', stacklevel=1)

            tc_arr[i] = self.get_key('ADAT', f'THICK{ka}')
            
            # add special logic for cylinders -- dynstall dies otherwise
            if np.isclose(tc_arr[i], 100):
                pcs = np.array([[-180, 0, 0.6, 0],
                                [180, 0, 0.6, 0]])
            
            else:
                shape = (self.get_key('ADAT', f'NALPHA{ka}')[0,0], 4)
                pcs = np.ndarray(shape, dtype=_DTYPE)
                pcs[:,0] = self.get_key('ADAT', f'ALPHA{ka}')[0,:]
                pcs[:,1] = self.get_key('ADAT', f'CL{ka}')[0,:]
                pcs[:,2] = self.get_key('ADAT', f'CD{ka}')[0,:]
                pcs[:,3] = self.get_key('ADAT', f'CM{ka}')[0,:]
            
            pc_lst.append(pcs)
            
        # in BGEOM the airfoil number is referred to and not the thickness
        # sort properly on thickness
        isort = tc_arr.argsort()
        pc.pc_sets = {1:(tc_arr[isort], [pc_lst[k] for k in isort])}

        return pc

    def get_ae(self):
        # BGEOMMB defines for each element the start and end point. We only
        # need half that of course.
        chord = self.get_key('BGEOMMB', 'CHORD')[0,0::2]
        # FIXME: is it radius or curved lenght
        radius = self.get_key('BGEOMMB', 'RJ')[0,0::2]
        thickness = self.get_key('BGEOMMB', 'BTHICK')[0,0::2]
        # FIXME: multiple pc sets can be defined but are ignored for now.
        pc_set_id = np.ones(chord.shape)

        ae = AEFile()
        ae.add_set(radius, chord, thickness, pc_set_id, set_id=None)

        return ae

    def get_blade_c2_def(self):

        # BLADENAME
        # DISTMODE	RJ
        # NBE	100
        # FOIL
        # MOVING

        # CE_X
        # CE_Y
        diam = self.get_key('RCON', 'DIAM')[0,0]
        cone = self.get_key('RCON', 'CONE')[0,0]
        tilt = self.get_key('RCON', 'TILT')[0,0]

        # in BLADED all is wrt LE, so we need chord
        chord = self.get_key('BGEOMMB', 'CHORD')[0,0::2]

        # c2_def = htc['new_htc_structure'].get_subsection_by_name('blade1')['c2_def']
        twist = self.get_key('BGEOMMB', 'TWIST')[0,0::2]

        # FIXME: what is the difference between RJ and DIST?
        # radius can be given eiter as curved length or radius
        # Distance: This can be entered as a distance along the blade or as a
        # distance along the blade root Z-axis. Select the appropriate option
        # at the base of the screen.
        dist = self.get_key('BGEOMMB', 'DIST')[0,0::2]
        rj = self.get_key('BGEOMMB', 'RJ')[0,0::2]
        # reference frame is EC, that seems to be indicated on the plots of the
        # blade rotor coordinates/pitch/axis tc (fig 4-1 and 4-2)
        ref_x = self.get_key('BGEOMMB', 'REF_X')[0,0::2]
        ref_y = self.get_key('BGEOMMB', 'REF_Y')[0,0::2]
        # EC wrt to LE in percentage of chord
        ce_x = self.get_key('BGEOMMB', 'CE_X')[0,0::2]
        ce_y = self.get_key('BGEOMMB', 'CE_Y')[0,0::2]

        # HAWC2 variables
        theta = twist*-1
        # theta_d = theta*180/np.pi
        ea_x = (50-ce_y)*0.01*chord
        ea_y = ce_x*0.01*chord

        # safe all c2_def positions in one array
        c2_arr = np.zeros((len(twist),4))
        c2_arr[:,0] = -1*(ref_y + ((ea_x*np.cos(theta) - ea_y*np.sin(theta))))
        c2_arr[:,1] =    (ref_x - ((ea_x*np.sin(theta) + ea_y*np.cos(theta))))
        # assume DIST is curved length
        c2_arr[:,2] = rj #dist
        c2_arr[:,3] = theta*180/np.pi

        return c2_arr

    def get_blade_st(self):
        # BMASSMB
        # BSTIFFMB

        # in HAWC2 structural pitch is relative to the twist
        twist = self.get_key('BGEOMMB', 'TWIST')[0,0::2]
        dist = self.get_key('BGEOMMB', 'DIST')[0,0::2]
        chord = self.get_key('BGEOMMB', 'CHORD')[0,0::2]
        thickness = self.get_key('BGEOMMB', 'BTHICK')[0,0::2]

        # CG location, in percentage chord wrt LE
        cm_x = self.get_key('BMASSMB', 'CM_X')[0,0::2]
        cm_y = self.get_key('BMASSMB', 'CM_Y')[0,0::2]
        mass = self.get_key('BMASSMB', 'MASS')[0,0::2]
        siner = self.get_key('BMASSMB', 'SINER')[0,0::2]

        # Radii of gyration ratio: the radius of gyration of mass about y_m
        # divided by the radius of gyration of mass about x_m. This defaults to
        # the relative profile thickness but can be over-written by un-checking
        # the “Use default radii of gyration ratio” checkbox.
        rgratio = self.get_key('BMASSMB', 'RGRATIO')[0,0::2]
        # FIXME: DEF_RGRATIO but which value belongs to which option?
        # assume 0 means "default is off"
        def_rgratio = self.get_key('BMASSMB', 'DEF_RGRATIO')[0,0]

        # FIXME: BETA_M: figure 3-2 from v4.8: mass axis orientation, radians?
        beta_m = self.get_key('BMASSMB', 'BETA_M')[0,0::2]
        # Mass axis orientation: the orientation of the principle axis of inertia.
        # This defaults to the orientation of aerodynamic twist, but can be
        # over-written by un-checking the “Use default mass axis orientation”
        # checkbox. (See diagram below)
        # FIXME: DEF_BETA_M but which value belongs to which option?
        # assume 0 means "default is off"
        def_beta_m = self.get_key('BMASSMB', 'DEF_BETA_M')[0,0]

        # DEF_RGRATIO	0
        # DEF_BETA_M	0

        # RADPOS
        # PITCHPOS

        # DEF_RGRATIO	0
        # DEF_BETA_M	0
        # ICEDBLADES	 0, 0, 0
        # ICEDENSITY	 700
        # TIPCHORD	 0
        # NPOINT	3
        # RADPOS	 .2, 30.12345, 40.12345
        # PITCHPOS	 .2, 30.1, 40.1
        # PM_X	 0, 0, 0
        # PM_Y	 20, 10, 20
        # PMASS	 1000, 30, 1

        # elastic axis, in percentage chord wrt LE
        ce_x = self.get_key('BGEOMMB', 'CE_X')[0,0::2]
        ce_y = self.get_key('BGEOMMB', 'CE_Y')[0,0::2]

        # 3.5 Blade stiffness distribution (page 19)
        # The stiffness must be defined about the principal axis of inertia at
        # each blade station (see 3.1). The stiffness is the product of Young’s
        # Modulus for the material and the second moment of area for the xp or
        # yp directions as appropriate. The principal axis orientation is defined
        # as an input, and defaults to the aerodynamic twist. In this case it is
        # assumed to be parallel and perpendicular to the chord line. If the
        # principal axis orientation is different from the aerodynamic twist, click
        # the Use default principal axis orientation to off. (see diagram below)
        eiflap = self.get_key('BSTIFFMB', 'EIFLAP')[0,0::2]
        eiedge = self.get_key('BSTIFFMB', 'EIEDGE')[0,0::2]
        beta_s = self.get_key('BSTIFFMB', 'BETA_S')[0,0::2]
        # gj = self.get_key('BSTIFFMB', 'GJ')[0,0::2]
        # cs_x = self.get_key('BSTIFFMB', 'CS_X')[0,0::2]
        # cs_y = self.get_key('BSTIFFMB', 'CS_Y')[0,0::2]
        # gaflap = self.get_key('BSTIFFMB', 'GAFLAP')[0,0::2]
        # gaedge = self.get_key('BSTIFFMB', 'GAEDGE')[0,0::2]
        # FIXME: DEF_BETA_S probably says what the frame of reference is
        # assume 0 means "default is off"
        # self.get_key('BMASSMB', 'DEF_BETA_S')[0,0::2]

        # for plotting purposes
        sel = beta_m < -1
        beta_m[sel] += np.pi
        sel = beta_s < -1
        beta_s[sel] += np.pi

        # calculate the curved length
        # c2_arr = self.get_blade_c2_def()
        # curved_len_arr = curved_length(c2_arr)

        ea_x = (50-ce_y)*0.01*chord
        ea_y = ce_x*0.01*chord
        cg_x = (50-cm_y)*0.01*chord
        cg_y = cm_x*0.01*chord

        # mass moments
        Ipm_y = siner + (mass*(cg_x-ea_x)**2)
        ri_y = np.sqrt( (Ipm_y / (1+rgratio**2)) / mass )
        ri_x = rgratio * ri_y
        # in BLADED rgratio = y/x, but in HAWC2 it x=y so we swap them

        # just choose E, G so the rest follows
        E = 1e10
        G = 1e17  # rigid in torsion, shear
        # however, we don't know EA (probably because stiff?), so assume A is
        # a full box based on chord length and maximum airfoil thickness
        # this will make the extensional stiffness EA quite large, but the user
        # would still end up with a physical meaningful value
        A = np.ones_like(dist)  # assume A is unity
        J = np.ones_like(dist)  # dummy value for J
        gx = 0.5 * A  # shear factor, dummy of 0.5
        gy = 0.5 * A  # shear factor, dummy of 0.5
        sc_x = ea_x.copy()  # assume shear center collocated with elastic center
        sc_y = ea_x.copy()  # assume shear center collocated with elastic center

        starr = np.ndarray((len(dist),19))
        starr[:,0] = dist
        starr[:,1] = mass
        starr[:,2] = cg_x
        starr[:,3] = cg_y
        # radius of gyration
        starr[:,4] = ri_x
        starr[:,5] = ri_y
        # shear center
        starr[:,6] = sc_x
        starr[:,7] = sc_y
        # E, G: choose
        starr[:,8] = E
        starr[:,9] = G
        starr[:,10] = eiflap/E
        starr[:,11] = eiedge/E
        starr[:,12] = J
        # shear factor
        starr[:,13] = gx
        starr[:,14] = gy
        starr[:,15] = A
        starr[:,16] = -1*(beta_s - twist) # structural pitch
        starr[:,17] = ea_x
        starr[:,18] = ea_y

        st = StFile()
        st.main_data_sets = {1:{1:starr}}
        # st.cols = wetb.hawc2.st_file.stc.split()
        return st

    def get_flange_masses(self):
        """Get locations, masses of tower flanges.
        Return nodes and z-offsets (tower coordinate system) for each flange.
        """
        
        tower_stations = self.get_key('TGEOM', 'TJ')[0]
        flange_zs = self.get_key('TMASS', 'HTPM')[0]
        flange_ms = self.get_key('TMASS', 'MTPM')[0]
        
        nodes= []
        off_zs = []
        for flange_z, flange_m in zip(flange_zs, flange_ms):
            node = np.argmax(flange_z < tower_stations)
            off_z = -(flange_z - tower_stations[node - 1])
            nodes.append(node)
            off_zs.append(off_z)
        
        return flange_zs, flange_ms, nodes, off_zs

    def get_tower_c2_def_st(self):
        """Ignores nodes that are not at the centerline

        Returns
        -------
        c2_arr : TYPE
            DESCRIPTION.
        st : TYPE
            DESCRIPTION.

        """

        # save all c2_def positions in one array
        tower_stations = self.get_key('TGEOM', 'TJ')  # tower stations [m]
        t_nodes = np.zeros((tower_stations.size, 3))
        t_nodes[:, 2] = tower_stations
        c2_arr = np.zeros((t_nodes.shape[0],4))
        c2_arr[:,0:2] = t_nodes[:,:-1].copy()
        c2_arr[:,2] = -t_nodes[:,-1] + t_nodes[0,-1] # It is minus for HAWC2 model
        c2_arr[:,3] = 0.0
                
        # material props from prj file
        material = self.get_key('TMASS', 'MATERIAL')[0]
        name_mat = np.unique(material)
        mat_prop = {}
        for i in name_mat:
            mat_prop[i] = self.get_key('TMASS', "%s"%i) # rho, E, G
            if mat_prop[i][0][0] <= 0:
                warnings.warn('Tower density is not positive! Setting to 8050.')
                mat_prop[i][0][0] = 8050
            if mat_prop[i][0][1] <= 0:
                warnings.warn('Tower E is not positive! Setting to 2.1E+18.')
                mat_prop[i][0][1] = 2.1E18
            if mat_prop[i][0][2] <= 0:
                warnings.warn('Tower G is not positive! Setting to 7.93E17.')
                mat_prop[i][0][2] = 7.93E17
        t_mat = np.array([mat_prop[i][0] for i in material])  # tower material array [nstn x 3]
        
        # dimensions from TMASS
        t_diam = self.get_key('TGEOM', 'TDIAM')  # tower diameter [m]
        t_thick = self.get_key('TMASS', 'WALLTHICK')  # wall thickness [mm]
        
        # get the inertial values
        d1 = t_diam.copy()  # outer diameter [m]
        t = t_thick.copy() * 1e-3  # thickness [m]
        rho = t_mat[:, 0]  # density [kg/m3]
        m, A, I_g, I_m = get_circular_inertia(d1, t, rho)
        
        # compare calculated mass to TOWM
        t_m = self.get_key('TMASS', 'TOWM')
        np.testing.assert_allclose(m, t_m, atol=0.1)
               

        # FIXME: MAYBE ANOTHER FIX FOR DIFFERENT MATERIALS
        # return c2_arr
        starr = np.zeros((c2_arr.shape[0], 19))
        starr[:,0] = -c2_arr[:,2] #
        starr[:,1] = m
        starr[:,2] = 0.0 # no cg offset
        starr[:,3] = 0.0 # no cg offset
        # radius of gyration
        starr[:,4] = np.sqrt(I_m/m) # ri_x = (I_m/m)**0.5 = (I_g/A)**0.5
        starr[:,5] = np.sqrt(I_m/m) # ri_y = (I_m/m)**0.5 = (I_g/A)**0.5
        # shear center
        starr[:,6] = 0.0 # no shear center offset
        starr[:,7] = 0.0 # no shear center offset
        # E, G: choose
        starr[:,8] = t_mat[:, 1]
        starr[:,9] = t_mat[:, 2]
        starr[:,10] = I_g
        starr[:,11] = I_g
        starr[:,12] = I_g*2
        # shear factor
        starr[:,13] = 3/4 # for circular section check it again
        starr[:,14] = 3/4 # for circular section check it again
        starr[:,15] = A
        starr[:,16] = 0.0 # structural pitch
        starr[:,17] = 0.0
        starr[:,18] = 0.0

        st = StFile()
        st.main_data_sets = {1:{1:starr}}

        return c2_arr, st

    def get_hub(self):
        # Blade root length, in blade root coordinates (with tilt and cone)
        # see figure 4-1 in Blade User Manual 4.8
        root = self.get_key('RCON', 'ROOT')[0,0]

        # Hub mass: the mass of the hub, including the spinner and any blade
        #   root section.
        # Hub mass centre: the distance from the intersection of the shaft and
        #   blade axes to the centre of mass of the hub, in a direction measured
        #   away from the tower.
        # Moments of inertia: the moment of inertia of the hub mass about the
        #   shaft axis must be defined.
        # The inertia about an axis perpendicular to the shaft may also be
        #   entered with its origin about the hub centre of mass.

        hubmas = self.get_key('RMASS', 'HUBMAS')[0,0]
        hubine = self.get_key('RMASS', 'HUBINE')[0,0]
        # FIXME: which axis is this? Would that translate to both xx and yy in H2?
        hubine2 = self.get_key('RMASS', 'HUBINE2')[0,0]

        # Enter the Spinner diameter. This is the diameter of any spinner or
        # nose-cone, within which the blades themselves experience no
        # aerodynamic forces.
        spind = self.get_key('RCON', 'SPIND')[0,0]

        cmass_hub = {'x':0, 'y':0, 'z':0, 'm':hubmas,
                     'Ixx':0, 'Iyy':0, 'Izz':hubine}

        return cmass_hub, root

    def get_drivetrain(self, len_shaft):

        # TODO: brake
        # MSTART BRAKE

        # DTRAIN
        # dtrain = self.get_key('DTRAIN')
        ginert = self.get_key('DTRAIN', 'GINERT')[0,0] # generator inertia
        gratio = self.get_key('DTRAIN', 'GRATIO')[0,0]
        # The additional inertia of the high speed shaft may also be specified
        # along with the inertia of the gearbox which is referred to the HSS.
        # gbxinert = self.get_key('DTRAIN', 'GBXINERT')[0,0] # gearbox inertia
        # FIXME: brake position?
        bpos = self.get_key('DTRAIN', 'BPOS')[0,0]
        # LSS seems to have a DOF indicator in LSSDOF, and has 6 elements
        # 4th element is torsion it seems
        klss = self.get_key('DTRAIN', 'KLSS')[0,3]
        # FIXME: what does this damping value mean? Simply C in the dyn system
        # theta_dt_dt*I + C*theta_dt + K*theta = 0 ?
        dlss = self.get_key('DTRAIN', 'DLSS')[0,3]
        # HSS only has one DOF
        khss = self.get_key('DTRAIN', 'KHSS')[0,0]
        dhss = self.get_key('DTRAIN', 'DHSS')[0,0]

        # Convert torsional stiffness definition into a beam stiffness
        # that applies for the considered shaft length (c2_def)
        G = 80e9 # typical steel value
        Ip = klss*len_shaft/G
        starr = np.zeros((2,19))
        starr[:,0] = [0, len_shaft] # st file is always normalised
        starr[:,1] = 0.5 # very light but not too light
        starr[:,2] = 0.0 # no cg offset
        starr[:,3] = 0.0 # no cg offset
        # radius of gyration: very light but not too light
        starr[:,4] = 0.001 #ri_x
        starr[:,5] = 0.001 #ri_y
        # shear center
        starr[:,6] = 0.0 # no shear center offset
        starr[:,7] = 0.0 # no shear center offset
        # E, G: choose
        starr[:,8] = 1e18
        starr[:,9] = G
        starr[:,10] = Ip/2 # area inertia's
        starr[:,11] = Ip/2
        starr[:,12] = Ip
        # shear factor
        starr[:,13] = 1000 # stiff in shear
        starr[:,14] = 1000
        starr[:,15] = 10 # cross-section area
        starr[:,16] = 0.0 # structural pitch
        starr[:,17] = 0.0 # elastic axis
        starr[:,18] = 0.0
        st = StFile()
        st.main_data_sets = {1:{1:starr}}

        # FIXME: INERTIATOCLUTCH??
        # inertiatoclutch = self.get_key('DTRAIN', 'INERTIATOCLUTCH')

        # FIXME: what is the location of these inertia's?
        # FIXME: what is the mass? It seems the mass is put elsewhere?

        # cmass_gbx = {'Izz':Izz}
        # cmass_gen = {'x':x, 'y':y, 'z':z, 'm':nacmas,
        #              'Ixx':Ixx, 'Iyy':Iyy, 'Izz':Izz}

        # convert HSS inertia to LSS
        gen_iner_lss = ginert * gratio * gratio
        cmass_gen = {'x':0, 'y':0, 'z':0, 'm':0,'Ixx':0, 'Iyy':0, 'Izz':gen_iner_lss}

        return cmass_gen, st

    def get_nacelle(self):

        # also get the nacell concentrated mass
        nacmas = self.get_key('NMASS', 'NACMAS')[0,0]
        # includes the nacelle structure and all the machinery within it.
        # It does not include the rotor blades and hub. If the Direct Drive
        # (see 4.1) option is selected, the mass of the generator (see 4.2)
        # is also excluded.

        # position of cg relative to tower top, nacelle mass coord system
        # based on figure 4-7 of bladed manual 4.8
        # FIXME: is this correct? z means SS, and at the edge of the nacelle?
        nmx = self.get_key('NMASS', 'NMX')[0,0] # width?
        nmy = self.get_key('NMASS', 'NMY')[0,0] # length?
        nmz = self.get_key('NMASS', 'NMZ')[0,0] # height?
        iyaw = self.get_key('NMASS', 'IYAW')[0,0]
        inod = self.get_key('NMASS', 'INOD')[0,0]
        iroll = self.get_key('NMASS', 'IROLL')[0,0]

        # MSTART NGEOM
        # NACW	 x.000
        # NACL	 x.000
        # NACH	 x.000
        # NACZ	 0
        # NACCD	 x.0
        # NAERO	N

        # height of the nacelle/towertop: the BLADED model does not have to
        # have consistent shaft length with tilt angle arriving at the rotor
        # center, so adjust towertop length in HAWC2 accordingly

        # Tower height: from the ground or sea surface to the yaw bearing
        # (only needed if the tower itself has not been defined)
        towht = self.get_key('RCON', 'TOWHT')[0,0]

        # Hub height: from the ground to the centre of the rotor
        # (i.e. the intersection of the blade and shaft axes)
        height = self.get_key('RCON', 'HEIGHT')[0,0]

        # horizontal distance between rotor center and tower center line
        ovrhng = self.get_key('RCON', 'OVRHNG')[0,0]
        tilt = self.get_key('RCON', 'TILT')[0,0]

        # FOR HAWC2, WE HAVE
        len_shaft = ovrhng / np.cos(tilt)
        len_towertop = height - towht - len_shaft*np.sin(tilt)
        assert len_towertop > 0
        # assert len_towertop + len_shaft*np.sin(tilt) + towht - height == 0

        # concentrated mass in HAWC2
        # 1  : nodenr
        # 2-4: x, y, z offset
        # 5  : mass
        # 6-8: Ixx, Iyy, Izz
        # 9-11: Ixy, Ixz, Iyz (optional)
        # in HAWC2 we would be in the tower top coordinate system, and is
        # usually the same as the global coordinate system
        x = nmx
        y = -nmy # move upwind wrt tower
        z = -nmz # move upward
        Ixx = inod
        Iyy = iroll
        Izz = iyaw

        cmass_nacelle = {'x': x, 'y': y, 'z': z, 'm': nacmas,
                         'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz}

        return cmass_nacelle, len_shaft, len_towertop

    def get_towershadow_drag(self):
        """Get the aerodrag values"""

        tstart, tend = self.get_key('TGEOM', 'TJ')[0][[0, -1]]
        dbottom, dtop = self.get_key('TGEOM', 'TDIAM')[0][[0, -1]]

        aerodrag = np.zeros((2, 3))
        aerodrag[:, 0] = [tstart, tend]  # tower bottom and top
        aerodrag[:, 1] = 0.6  # constant 0.6 drag for entire tower
        aerodrag[:, 2] = [dbottom, dtop]  # tower bottom and top

        return aerodrag

    def get_control(self):

        # MSTART CONTROL

        # MSTART PCOEFF
        # TSRMIN	 x
        # TSRMAX	 x
        # TSRSTP	 x.x
        # PITCH	-x.0E-00
        # PITCH_END	-x.0E-00
        # PITCH_STEP	 0
        # OMEGA	 x.654321

        # MSTART IDLING

        # AEROINFO

        # pitch actuator rate limits
        pitch_actuator = self.xmlroot.PitchActuator
        if pitch_actuator.SetpointTrajectory.HasRateLimits:
            # pitch_actuator.SetpointTrajectory.LowerRateLimit
            # pitch_actuator.SetpointTrajectory.UpperRateLimit
            print(pitch_actuator.SetpointTrajectory.UpperRateLimit*30/np.pi)

        if pitch_actuator.SetpointTrajectory.HasAccelerationLimits:
            # pitch_actuator.SetpointTrajectory.LowerAccelerationLimit
            # pitch_actuator.SetpointTrajectory.UpperAccelerationLimit
            print(pitch_actuator.SetpointTrajectory.UpperAccelerationLimit*30/np.pi)

        # pitch_actuator.PositionResponse.values() # 1st or 2nd order
        # pitch_actuator.PositionResponse.Frequency
        # pitch_actuator.PositionResponse.Damping

        # pitch_actuator.RateResponse.values() # 1st or 2nd order
        # pitch_actuator.RateResponse.LagTime
        # pitch_actuator.PositionResponse.Damping

        # and more: bearing friction, etc

        return

    def convert(self, fname_tmpl, fname_htc, modelpath=None, save=True):

        fname_prj = os.path.basename(fname_htc).replace('.htc', '')

        # for convenience, we start from an htc template that otherwise has the
        # right structure, but has only 2 nodes for each body.
        # assume it is in the same folder as where the HTC will be written to
        basepath = os.path.dirname(modelpath)
        htc = HTCFile(fname_tmpl, modelpath=modelpath)

        # add the relevant body/struc commands
        htc.new_htc_structure.add_line(' ', [],
                                       comments='struct_inertia_output_file_name ' +
                                       f'bodyeig/{fname_prj}_struc_inertia.dat;')
        htc.new_htc_structure.add_line(' ', [],
                                       comments='structure_eigenanalysis_file_name ' +
                                       f'bodyeig/{fname_prj}_struc_eig.dat;')
        htc.new_htc_structure.add_line(' ', [],
                                       comments='body_eigenanalysis_file_name ' +
                                       f'bodyeig/{fname_prj}_body_eig.dat;')

        # cone = prj.get_key('RCON', 'CONE')
        # Lateral offset: the horizontal offset between the shaft and tower axes.

        # Brake
        # steps = prj.get_key('BRAKE', 'STEPS')
        # TORQUE = prj.get_key('BRAKE', 'TORQUE')
        # TIME = prj.get_key('BRAKE', 'TIME')

        # -------------------------------------------------------------------------
        # BLADES
        # extract blade geometry, and add to c2_def section in the htc file
        c2_blade = self.get_blade_c2_def()
        htc = add_c2_arr_htc(htc, c2_blade, 'blade1')
        
        # -------------------------------------------------------------------------
        # NACELLE
        cmass_nacelle, len_shaft, len_towertop = self.get_nacelle()
        # set towertop length
        # nacelle = htc['new_htc_structure'].get_subsection_by_name('towertop')
        c2_tt = np.zeros((2,4))
        c2_tt[1,2] = -len_towertop
        htc = add_c2_arr_htc(htc, c2_tt, 'towertop')

        # -------------------------------------------------------------------------
        # attach nacelle mass/inertia to towertop, first node (yaw bearing)
        towertop = htc['new_htc_structure'].get_subsection_by_name('towertop')
        values = [v for k,v in cmass_nacelle.items()]
        cmass = towertop.add_line('concentrated_mass', [1] + values)
        key = cmass.location().split('/')[-1]
        towertop[key].comments = 'nacelle mass (inc rotating) and inertia (non-rotating)'

        # -------------------------------------------------------------------------
        # set shaft length
        c2_s = np.zeros((2,4))
        c2_s[1,2] = len_shaft
        htc = add_c2_arr_htc(htc, c2_s, 'shaft')

        # -------------------------------------------------------------------------
        # generator's, shaft torsional flexibility
        cmass_gen, st_shaft = self.get_drivetrain(len_shaft)
        st_shaft.save(pjoin(basepath, f'data/{fname_prj}_shaft.st'))

        shaft = htc['new_htc_structure'].get_subsection_by_name('shaft')
        # LSS inertia
        # values = [v for k,v in cmass_lss.items()]
        # cmass = shaft.add_line('concentrated_mass', [2] + values)
        # key = cmass.location().split('/')[-1]
        # shaft[key].comments = 'inertia LSS'
        # Inertia of the HSS is already expressed wrt LSS
        values = [v for k,v in cmass_gen.items()]
        cmass = shaft.add_line('concentrated_mass', [1] + values)
        gratio = self.get_key('DTRAIN', 'GRATIO')[0, 0]
        key = cmass.location().split('/')[-1]
        shaft[key].comments = f'inertia HSS, expressed in LSS, GBR={gratio}'

        # -------------------------------------------------------------------------
        # set tilt angle
        tilt = self.get_key('RCON', 'TILT')[0,0]
        ori = htc['new_htc_structure'].orientation
        # tilt is on the 2nd relative block, and set angle on 2nd set of eulerang
        # check comments of the htc file:
        # print(ori['relative__2']['mbdy2_eulerang__2'].comments)
        ori['relative__2']['mbdy2_eulerang__2'].values = [tilt*180/np.pi, 0, 0]
        ori['relative__2']['mbdy2_eulerang__2'].comments = 'tilt angle'

        # coning angle
        cone = self.get_key('RCON', 'CONE')[0,0]
        # coning is on 3, 4 and 5 (all hub orientations)
        for k in range(3,6):
            # print(ori[f'relative__{k}']['mbdy2_eulerang__3'].comments)
            ori[f'relative__{k}']['mbdy2_eulerang__3'].values = [-cone*180/np.pi, 0, 0]
            ori[f'relative__{k}']['mbdy2_eulerang__3'].comments = 'cone angle'

        # TODO: blade mounting sweep and cone as well??

        # -------------------------------------------------------------------------
        # HUB mass as concentrated mass on the shaft end
        # FIXME: make sure we can get the hub root loads including these inertia's
        cmass_hub, hublen = self.get_hub()
        values = [v for k,v in cmass_hub.items()]
        cmass = shaft.add_line('concentrated_mass', [2] + values)
        key = cmass.location().split('/')[-1]
        shaft[key].comments = 'Hub mass and inertia'

        # -------------------------------------------------------------------------
        # hub length
        c2_hub = np.zeros((2,4))
        c2_hub[1,2] = hublen
        htc = add_c2_arr_htc(htc, c2_hub, 'hub1')

        # -------------------------------------------------------------------------
        # aerodrag and tower shadow
        
        # tower aerodrag
        twr_aerodrag = self.get_towershadow_drag()
        htc.aerodrag.aerodrag_element.sec__1.values = twr_aerodrag[0, :]
        htc.aerodrag.aerodrag_element.sec__2.values = twr_aerodrag[1, :]
        
        # tower shadow
        htc.wind.tower_shadow_potential_2.radius__1.values = [twr_aerodrag[0, 0],
                                                              twr_aerodrag[0, 2]/2]
        htc.wind.tower_shadow_potential_2.radius__2.values = [twr_aerodrag[1, 0],
                                                              twr_aerodrag[1, 2]/2]
        
        # nacelle aerodrag
        htc.aerodrag.aerodrag_element__2.sec__1.values = [0, 0.8, 5]
        htc.aerodrag.aerodrag_element__2.sec__2.values = [len_shaft, 0.8, 5]

        # -------------------------------------------------------------------------
        # aero settings
        
        htc.aero.induction_method.values = [1]
        htc.aero.aerosections.values = [20]

        # -------------------------------------------------------------------------
        # TOWER ST FILE and C2_DEF section
        # WARNING: only includes nodes that are at centerline, side-elements are
        # ignored
        # TODO: compare between TGEOM/TMASS and TSTIFF approaches, it seems that
        # the shear stiffness is not the same, see check_tower()
        c2_tow, st = self.get_tower_c2_def_st()
        st.save(pjoin(basepath, f'data/{fname_prj}_tower.st'))
        htc = add_c2_arr_htc(htc, c2_tow, 'tower')
        
        # add tower-flange point masses
        tower = htc['new_htc_structure'].get_subsection_by_name('tower')
        flange_zs, masses, nodes, off_zs = self.get_flange_masses()
        for flange_z, mass, node, off_z in zip(flange_zs, masses, nodes, off_zs):
            cmass = tower.add_line('concentrated_mass', [node, 0, 0, off_z, mass, 0, 0, 0])
            key = cmass.location().split('/')[-1]
            tower[key].comments = f'Flange mass, {mass} kg at {flange_z} m'

        # hub height
        htc.wind.center_pos0.values = [0, 0, -self.get_key('RCON', 'HEIGHT')[0,0]]

        # -------------------------------------------------------------------------
        # set damping in tower and blades
        
        blade = htc['new_htc_structure'].get_subsection_by_name('blade1')
        tower.damping_posdef = [0, 0, 0, 1.686E-3, 1.686E-3, 1e-1]
        blade.damping_posdef = [0, 0, 0, 9.8e-4, 1.5e-3, 1e-3]

        # -------------------------------------------------------------------------
        # convergence limits
        
        htc.simulation.convergence_limits = [100, 1, 1e-7]

        # -------------------------------------------------------------------------
        # CONTROLLER
        
        htc.continue_in_file = './htc/md70_control.htc'

        # -------------------------------------------------------------------------
        # update output channles
        
        sensors = htc.output.sensors
        nbodies_tower = tower.c2_def.nsec.values[0] - 1
        nbodies_blade = blade.c2_def.nsec.values[0] - 1
        for i, s in enumerate(sensors):
            if 'tower yaw bearing' in str(s):
                s.values[1:3] = [nbodies_tower, 2]
            elif '50% local' in str(s):
                s.values[1] = nbodies_blade // 2
            elif 'Tower top' in str(s):
                s.values[2:4] = [nbodies_tower, 1.0]
            elif 'tip pos' in str(s):
                s.values[2:4] = [nbodies_blade, 1.0]
        
        output = htc.output
        output.add_line(' ', [], comments='DLL CHANNELS')
        dlls = {1: [[1, 'Generator contactor [int]'],
                    [2, 'Demanded blade 1 pitch [rad]'],
                    [3, 'Demanded blade 2 pitch [rad]'],
                    [4, 'Demanded blade 3 pitch [rad]'],
                    [5, 'Collective demanded pit [rad]'],
                    [6, 'Collective demanded pit rate [rad/s]'],
                    [7, 'Demanded generator torque [Nm]'],
                    [8, 'Safety system to activate [int]']],
                2: [[1, 'Mgen LSS [Nm]'],
                    [2, 'Pelec    [W]'],
                    [3, 'Mframe   [Nm]'],
                    [4, 'Mgen HSS [Nm]'],
                    [8, 'Grid flag [0=run/1=stop]']]
                }
        for k, v in dlls.items():
            [output.add_line('dll inpvec',
                             [k, i, '#', s])
             for (i, s) in v]
        output.add_line(' ', [], comments='GENERAL')


        # -------------------------------------------------------------------------
        # set all file names correct
        tower = htc['new_htc_structure'].get_subsection_by_name('tower')
        tower.timoschenko_input.filename.values[0] = f'data/{fname_prj}_tower.st'

        shaft = htc['new_htc_structure'].get_subsection_by_name('shaft')
        shaft.timoschenko_input.filename.values[0] = f'data/{fname_prj}_shaft.st'

        blade1 = htc['new_htc_structure'].get_subsection_by_name('blade1')
        blade1.timoschenko_input.filename.values[0] = f'data/{fname_prj}_blade.st'

        for body_name in ['towertop', 'hub1']:
            body = htc['new_htc_structure'].get_subsection_by_name(body_name)
            body.timoschenko_input.filename.values[0] = 'data/template.st'

        htc['aero'].ae_filename.values[0] = f'data/{fname_prj}.ae'
        htc['aero'].pc_filename.values[0] = f'data/{fname_prj}.pc'
        htc.simulation.logfile = f'log/{fname_prj}.log'
        htc.output.filename = f'res/{fname_prj}'
        
        # -------------------------------------------------------------------------
        # make other files
        
        st = self.get_blade_st()
        pc = self.get_pc()
        ae = self.get_ae()        

        # save files if requested
        if save:
            
            # -------------------------------------------------------------------------
            # htc files
            htc.save(pjoin(basepath, f'{fname_htc}'))
    
            # -------------------------------------------------------------------------
            # other data files
    
            # Blade st file
            st.save(pjoin(basepath, f'data/{fname_prj}_blade.st'))
    
            # extract profile coefficients and save to pc file
            pc.save(pjoin(basepath, f'data/{fname_prj}.pc'))
    
            # extract aerodynamic layout, and safe to ae file
            ae.save(pjoin(basepath, f'data/{fname_prj}.ae'))
        
        return htc, st, pc, ae
