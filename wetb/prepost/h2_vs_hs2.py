# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:23:15 2015

@author: dave
"""




import os

import numpy as np
#import scipy.interpolate as interpolate
import pandas as pd
from matplotlib import pyplot as plt

from wetb.prepost import Simulations as sim
from wetb.prepost import dlcdefs
from wetb.prepost import hawcstab2 as hs2
from wetb.prepost import mplutils


class Configurations:
    # HAWC2
    eigenan = {'[eigen_analysis]':True, '[time stop]':0,
               '[t0]'            :0,     '[output]'  :False}
    control = {'[Free shaft rot]':True,  '[dll]'        :True,
               '[fixed_op]'      :False, '[fixed_shaft]':False,
               '[init_wr]'       :0.5,   '[pitch_bearing]':True}
    opt_h2 = {'[output]'         :True,  '[hs2]'        :False,
              '[hawc2_exe]'      :'hawc2-latest',
              '[Case folder]'    :'HAWC2', '[hawc2]'    :True}
    fix_op = {'[Free shaft rot]' :False, '[dll]'        :False,
              '[fixed_op]'       :True,  '[fixed_shaft]':False,
              '[init_wr]'        :0.5,   '[fixed_omega]':0.5,
              '[pitch_bearing]'  :False}
    # HAWCStab2
    opt_hs2 = {'[output]'         :False, '[hs2]'        :True,
               '[Free shaft rot]' :True,  '[dll]'        :False,
               '[fixed_op]'       :False, '[fixed_shaft]':False,
               '[init_wr]'        :0.5,   '[fixed_omega]':0.5,
               '[pitch_angle]'    :0.0,   '[hawc2_exe]'  :'hs2cmd-latest',
               '[Case folder]'    :'HAWCStab2', '[hawc2]':False,
               '[pitch_bearing]'  :True}

    # AERODYNAMIC MODELLING OPTIONS
    aero_simple = {'[aerocalc]':1, '[Induction]':0, '[tip_loss]':0,
                   '[Dyn stall]':0, '[t0]':100, '[time stop]':150}
    # when induction is on, especially the low wind speeds will need more time
    # to reach steady state in HAWC2 compared to when there is no induction.
    aero_full = {'[aerocalc]':1, '[Induction]':1, '[tip_loss]':1,
                 '[Dyn stall]':0, '[t0]':700, '[time stop]':750}
    aero_ind = {'[aerocalc]':1, '[Induction]':1, '[tip_loss]':0,
                '[Dyn stall]':0, '[t0]':700, '[time stop]':750}

    blade_stiff_pitchC4 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                           '[blade_damp_z]':0.01, '[blade_set]':4,
                           '[blade_subset]':1, '[blade_posx]':-0.75}

    blade_stiff_pitchC2 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                           '[blade_damp_z]':0.01, '[blade_set]':4,
                           '[blade_subset]':1, '[blade_posx]':0.0}

    blade_stiff_pitch3C4 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                           '[blade_damp_z]':0.01, '[blade_set]':4,
                           '[blade_subset]':1, '[blade_posx]':0.75}

    blade_flex50_tstiff_C14 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                               '[blade_damp_z]':0.03, '[blade_set]':1,
                               '[blade_subset]':15, '[blade_posx]':-0.75,
                               '[blade_nbodies]': 17,
                               '[c12]':False, '[c14]':True}
    blade_flex50_tstiff_C12 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                               '[blade_damp_z]':0.03, '[blade_set]':1,
                               '[blade_subset]':13, '[blade_posx]':0.0,
                               '[blade_nbodies]': 17,
                               '[c12]':True, '[c14]':False}
    blade_flex50_C14 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                        '[blade_damp_z]':0.03, '[blade_set]':1,
                        '[blade_subset]':17, '[blade_posx]':-0.75,
                        '[blade_nbodies]': 17,
                        '[c12]':False, '[c14]':True}
    blade_flex50_C12 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                        '[blade_damp_z]':0.03, '[blade_set]':1,
                        '[blade_subset]':16, '[blade_posx]':0.0,
                        '[blade_nbodies]': 17,
                        '[c12]':True, '[c14]':False}
    blade_flex89_tstiff_C14 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':23, '[blade_posx]':-0.75,
                            '[blade_nbodies]': 17,
                            '[c12]':False, '[c14]':True}
    blade_flex89_tstiff_C12 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':22, '[blade_posx]':0.0,
                            '[blade_nbodies]': 17,
                            '[c12]':True, '[c14]':False}
    blade_flex89_t50_C14 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':19, '[blade_posx]':-0.75,
                            '[blade_nbodies]': 17,
                            '[c12]':False, '[c14]':True}
    blade_flex89_t50_C12 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':18, '[blade_posx]':0.0,
                            '[blade_nbodies]': 17,
                            '[c12]':True, '[c14]':False}
    blade_flex89_t50_C12_allstC14 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':21, '[blade_posx]':-0.75,
                            '[blade_nbodies]': 17,
                            '[c12]':False, '[c14]':True}
    blade_flex89_t50_C12_allstC12 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                            '[blade_damp_z]':0.03, '[blade_set]':1,
                            '[blade_subset]':19, '[blade_posx]':0.0,
                            '[blade_nbodies]': 17,
                            '[c12]':True, '[c14]':False}

    blade_flex89_t50_C12_cgshC14_eaC12 = {'[blade_damp_x]':0.03,
                                          '[blade_damp_y]':0.03,
                                          '[blade_damp_z]':0.03,
                                          '[blade_set]'   :1,
                                          '[blade_subset]':24,
                                          '[blade_posx]'  :0.0,
                                          '[blade_nbodies]': 17,
                                          '[c12]':True, '[c14]':False}

    blade_flex = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                  '[blade_damp_z]':0.01, '[blade_set]':1,
                  '[blade_subset]':1, '[blade_posx]':-0.75}

    blade_flex_allac = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                        '[blade_damp_z]':0.01, '[blade_set]':1,
                        '[blade_subset]':2, '[blade_posx]':-0.75}

    blade_flex_allac_11 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                           '[blade_damp_z]':0.01, '[blade_set]':1,
                           '[blade_subset]':7, '[blade_posx]':-0.75}

    blade_flex_allac_33 = {'[blade_damp_x]':0.02, '[blade_damp_y]':0.02,
                           '[blade_damp_z]':0.02, '[blade_set]':1,
                           '[blade_subset]':8, '[blade_posx]':-0.75}

    blade_flex_allac_50 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                           '[blade_damp_z]':0.03, '[blade_set]':1,
                           '[blade_subset]':9, '[blade_posx]':-0.75}

    blade_flex_allac_50_pitchC2 = {'[blade_damp_x]':0.03, '[blade_damp_y]':0.03,
                                   '[blade_damp_z]':0.03, '[blade_set]':1,
                                   '[blade_subset]':9, '[blade_posx]':0.0}

    # configurations for the B-series (which has quite a few changes)
    # B0001
    stiff_pc14_cgsheac14 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                            '[blade_damp_z]':0.01, '[blade_set]':1,
                            '[blade_subset]':3,    '[blade_posx]':-0.75,
                            '[blade_nbodies]': 17,
                            '[st_file]'     :'blade_flex_rect.st',
                            '[c12]':False, '[c14]':True,
                            '[ae_tolrel]': 1e-7,
                            '[ae_itmax]' : 2000,
                            '[ae_1relax]': 0.0}
    # B0002
    flex_tstiff_pc14_cgsheac14 = {'[blade_damp_x]':0.13, '[blade_damp_y]':0.13,
                                  '[blade_damp_z]':0.15, '[blade_set]':1,
                                  '[blade_subset]':5,    '[blade_posx]':-0.75,
                                  '[blade_nbodies]': 17,
                                  '[st_file]'     :'blade_flex_rect.st',
                                  '[c12]':False, '[c14]':True,
                                  '[ae_tolrel]': 1e-7,
                                  '[ae_itmax]' : 2000,
                                  '[ae_1relax]': 0.0}
    # B0003
    flex_pc14_cgsheac14 = {'[blade_damp_x]':0.13, '[blade_damp_y]':0.13,
                           '[blade_damp_z]':0.15, '[blade_set]':1,
                           '[blade_subset]':6,    '[blade_posx]':-0.75,
                           '[blade_nbodies]': 17,
                           '[st_file]'     :'blade_flex_rect.st',
                           '[c12]':False, '[c14]':True,
                           '[ae_tolrel]': 1e-7,
                           '[ae_itmax]' : 2000,
                           '[ae_1relax]': 0.0}
    # B0004
    flex_pc12_cgsheac12 = {'[blade_damp_x]':0.15, '[blade_damp_y]':0.15,
                           '[blade_damp_z]':0.17, '[blade_set]':1,
                           '[blade_subset]':7,    '[blade_posx]':0.00,
                           '[blade_nbodies]': 17,
                           '[st_file]'     :'blade_flex_rect.st',
                           '[c12]':True, '[c14]':False,
                           '[ae_tolrel]': 1e-7,
                           '[ae_itmax]' : 2000,
                           '[ae_1relax]': 0.0}
    # B0005, B0006, B0007, B0008
    flex_pc12_cgshc14_eac12 = {'[blade_damp_x]':0.15, '[blade_damp_y]':0.15,
                               '[blade_damp_z]':0.17, '[blade_set]':1,
                               '[blade_subset]':8,    '[blade_posx]':0.00,
                               '[blade_nbodies]': 17,
                               '[st_file]'     :'blade_flex_rect.st',
                               '[c12]':True, '[c14]':False,
                               '[ae_tolrel]': 1e-7,
                               '[ae_itmax]' : 2000,
                               '[ae_1relax]': 0.0}
    # B0009
    stiff_pc12_cgsheac14 = {'[blade_damp_x]':0.01, '[blade_damp_y]':0.01,
                            '[blade_damp_z]':0.01, '[blade_set]':1,
                            '[blade_subset]':2,    '[blade_posx]':0.00,
                            '[blade_nbodies]': 17,
                            '[st_file]'     :'blade_flex_rect.st',
                            '[c12]':True, '[c14]':False,
                            '[ae_tolrel]': 1e-7,
                            '[ae_itmax]' : 2000,
                            '[ae_1relax]': 0.0}

    def __init__(self):
        pass

    def opt_tags_h2_eigenanalysis(self, basename):
        """Return opt_tags suitable for a standstill HAWC2 eigen analysis.
        """
        opt_tags = [self.opt_h2.copy()]
        opt_tags[0].update(self.eigenan.copy())
        opt_tags[0]['[Case id.]'] = '%s_hawc2_eigenanalysis' % basename
        opt_tags[0]['[blade_damp_x]'] = 0.0
        opt_tags[0]['[blade_damp_y]'] = 0.0
        opt_tags[0]['[blade_damp_z]'] = 0.0
        opt_tags[0]['[blade_nbodies]'] = 1
        opt_tags[0]['[Windspeed]'] = 0.0
        opt_tags[0]['[init_wr]'] = 0.0
        opt_tags[0]['[operational_data]'] = 'case-turbine2-empty.opt'

        return opt_tags

    def opt_tags_hs_structure_body_eigen(self, basename):
        """Return opt_tags suitable for a standstill HAWCStab2 body eigen
        analysis, at 0 RPM.
        """
        opt_tags = [self.opt_hs2.copy()]
        opt_tags[0]['[Case id.]'] = '%s_hawc2_eigenanalysis' % basename
        opt_tags[0]['[blade_damp_x]'] = 0.0
        opt_tags[0]['[blade_damp_y]'] = 0.0
        opt_tags[0]['[blade_damp_z]'] = 0.0
        opt_tags[0]['[blade_nbodies]'] = 1
        opt_tags[0]['[Windspeed]'] = 0.0
        opt_tags[0]['[init_wr]'] = 0.0
        opt_tags[0]['[fixed_omega]'] = 0.0
        opt_tags[0]['[operational_data]'] = 'case-turbine2-empty.opt'

        return opt_tags

    def opt_tags_hs2(self, basename):

        opt_tags = [self.opt_hs2.copy()]
        opt_tags[0]['[Case id.]'] = '%s_hawcstab2' % basename
        return opt_tags

    def set_hs2opdata(self, master, basename):
        """Load the HS2 operational data file and create opt_tags for HAWC2
        cases.

        Returns
        -------
        opt_tags : list of dicts
        """
        fpath = os.path.join(master.tags['[data_dir]'],
                             master.tags['[operational_data]'])
        hs2_res = hs2.results()
        hs2_res.load_operation(fpath)
        omegas = hs2_res.operation.rotorspeed_rpm.values*np.pi/30.0
        winds = hs2_res.operation.windspeed.values
        pitchs = hs2_res.operation.pitch_deg.values

        return self.set_opdata(winds, pitchs, omegas, basename=basename)

    def set_opdata(self, winds, pitchs, omegas, basename=None):
        """Return opt_tags for HAWC2 based on an HAWCStab2 operational data
        file.

        Parameters
        ----------

        winds : ndarray(n)
            wind speed for given operating point [m/s]

        pitchs : ndarray(n)
            pitch angle at given operating point [deg]

        omegas : ndarray(n)
            rotor speed at given operating point [rad/s]

        basename : str, default=None
            If not None, the [Case id.] tag is composed out of the basename,
            wind speed, pitch angle and rotor speed. If set to None, the
            [Case id.] tag is not set.

        Returns
        -------
        opt_tags : list of dicts
        """

        # the HAWC2 cases
        opt_tags = []
        for wind, pitch, omega in zip(winds, pitchs, omegas):
            opt_dict = {}
            opt_dict.update(self.opt_h2.copy())
            opt_dict.update(self.fix_op.copy())
            rpl = (basename, wind, pitch, omega)
            if basename is not None:
                tmp = '%s_%02.0fms_%04.01fdeg_%04.02frads_hawc2' % rpl
                opt_dict['[Case id.]'] = tmp
            opt_dict['[Windspeed]'] = wind
            opt_dict['[pitch_angle]'] = pitch
            opt_dict['[fixed_omega]'] = omega
            opt_dict['[init_wr]'] = omega
#            opt_dict['[t0]'] = int(2000.0/opt_dict['[Windspeed]']) # or 2000?
#            opt_dict['[time stop]'] = opt_dict['[t0]']+100
#            opt_dict['[time_stop]'] = opt_dict['[t0]']+100
            opt_tags.append(opt_dict.copy())
        return opt_tags


class Sims(object):

    def __init__(self, sim_id, P_MASTERFILE, MASTERFILE, P_SOURCE, P_RUN,
                 PROJECT, POST_DIR):
        """
        Create HtcMaster() object
        =========================

        the HtcMaster contains all the settings to start creating htc files.
        It holds the master file, server paths and more.

        The master.tags dictionary holds those tags who do not vary for different
        cases. Variable tags, i.e. tags who are a function of other variables
        or other tags, are defined in the function variable_tag_func().

        It is considered as good practice to define the default values for all
        the variable tags in the master_tags

        Members
        -------

        Returns
        -------

        """

        self.sim_id = sim_id
        self.P_MASTERFILE = P_MASTERFILE
        self.MASTERFILE = MASTERFILE
        self.P_SOURCE = P_SOURCE
        self.P_RUN = P_RUN
        self.PROJECT = PROJECT
        self.POST_DIR = POST_DIR

        # TODO: write a lot of logical tests for the tags!!
        # TODO: tests to check if the dirs are setup properly (ending slahses)
        # FIXME: some tags are still variable! Only static tags here that do
        # not depent on any other variable that can change
        self.master = sim.HtcMaster()
        self.set_tag_defaults()

    def _var_tag_func(self, master, case_id_short=False):
        """
        Function which updates HtcMaster.tags and returns an HtcMaster object

        Only use lower case characters for case_id since a hawc2 result and
        logfile are always in lower case characters. Simulations.prepare_launch
        will force the value of the tags as defined in master.output_dirs
        to lower case.

        BE CAREFULL: if you change a master tag that is used to dynamically
        calculate an other tag, that change will be propageted over all cases,
        for example:
        master.tags['tag1'] *= master.tags[tag2]*master.tags[tag3']
        it will accumlate over each new case. After 20 cases
        master.tags['tag1'] = (master.tags[tag2]*master.tags[tag3'])^20
        which is not wanted, you should do
        master.tags['tag1'] = tag1_base*master.tags[tag2]*master.tags[tag3']

        """

        mt = master.tags

        dlc_case = mt['[Case folder]']
        mt['[data_dir]'] = 'data/'
        mt['[res_dir]'] = 'res/%s/' % dlc_case
        mt['[log_dir]'] = 'logfiles/%s/' % dlc_case
        mt['[htc_dir]'] = 'htc/%s/' % dlc_case
        mt['[case_id]'] = mt['[Case id.]']
        mt['[DLC]'] = dlc_case
        mt['[pbs_out_dir]'] = 'pbs_out/%s/' % dlc_case
        mt['[pbs_in_dir]'] = 'pbs_in/%s/' % dlc_case
        mt['[iter_dir]'] = 'iter/%s/' % dlc_case

        if mt['[eigen_analysis]']:
            rpl = os.path.join(dlc_case, mt['[Case id.]'])
            mt['[eigenfreq_dir]'] = 'res_eigen/%s/' % rpl

        # for HAWCStab2 certain things have to be done differently
        if mt['[hs2]']:
            mt['[htc_dir]'] = ''
            mt['[t0]'] = 0
            mt['[time stop]'] = 1
            mt['[hawc2]'] = False
            mt['[output]'] = False
            mt['[copyback_files]'] = ['./*.ind', './*.pwr', './*.log',
                                      './*.cmb', './*.bea']
            mt['[copyback_frename]'] = [mt['[res_dir]'], mt['[res_dir]'],
                                        mt['[log_dir]'], mt['[res_dir]'],
                                        mt['[res_dir]']]
            if mt['[hs2_bladedeform_switch]']:
                mt['[hs2_bladedeform]'] = 'bladedeform'
            else:
                mt['[hs2_bladedeform]'] = 'nobladedeform'

            if int(mt['[tip_loss]']) == 1:
                mt['[hs2_tipcorrect]'] = 'tipcorrect'
            else:
                mt['[hs2_tipcorrect]'] = 'notipcorrect'

            if int(mt['[Induction]']) == 1:
                mt['[hs2_induction]'] = 'induction'
            else:
                mt['[hs2_induction]'] = 'noinduction'

            if mt['[hs2_gradients_switch]']:
                mt['[hs2_gradients]'] = 'gradients'
            else:
                mt['[hs2_gradients]'] = 'nogradients'

        mt['[windspeed]'] = mt['[Windspeed]']
        mt['[time_stop]'] = mt['[time stop]']
        mt['[duration]'] = str(float(mt['[time_stop]']) - float(mt['[t0]']))

        return master

    def _set_path_auto_config(self, verbose=True):
        """
        auto configure directories: assume you are running in the root of the
        relevant HAWC2 model
        and assume we are in a simulation case of a certain turbine/project
        """
        (self.P_RUN, self.P_SOURCE, self.PROJECT,
             self.sim_id, self.P_MASTERFILE,
             self.MASTERFILE, self.POST_DIR) = dlcdefs.configure_dirs(verbose=verbose)

    def _set_path_config(self, runmethod='here'):
        """
        Set the path configuration into the tags
        """

        self.runmethod = runmethod

        if runmethod == 'here':
            self._set_path_auto_config()
        elif runmethod in ['local', 'local-script', 'none', 'local-ram']:
            self.p_root = '/home/dave/SimResults/h2_vs_hs2/'
        elif runmethod == 'windows-script':
            self.p_root = '/mnt/D16731/dave/Documents/_SimResults'
        elif runmethod == 'gorm':
            self.p_root = '/mnt/hawc2sim/h2_vs_hs2'
        elif runmethod == 'jess':
            self.p_root = '/mnt/hawc2sim/h2_vs_hs2'
        else:
            msg='unsupported runmethod, options: none, local, gorm or opt'
            raise ValueError(msg)

        if not runmethod == 'here':
            self.P_RUN = os.path.join(self.p_root, self.PROJECT, self.sim_id)

        self.master.tags['[master_htc_file]'] = self.MASTERFILE
        self.master.tags['[master_htc_dir]'] = self.P_MASTERFILE
        # directory to data, htc, SOURCE DIR
        if self.P_SOURCE[-1] == os.sep:
            self.master.tags['[model_dir_local]']  = self.P_SOURCE
        else:
            self.master.tags['[model_dir_local]']  = self.P_SOURCE + os.sep
        if self.P_RUN[-1] == os.sep:
            self.master.tags['[run_dir]'] = self.P_RUN
        else:
            self.master.tags['[run_dir]'] = self.P_RUN + os.sep

        self.master.tags['[post_dir]'] = self.POST_DIR
        self.master.tags['[sim_id]'] = self.sim_id
        # set the model_zip tag to include the sim_id
        rpl = (self.PROJECT, self.master.tags['[sim_id]'])
        self.master.tags['[model_zip]'] = '%s_%s.zip' % rpl

    def set_tag_defaults(self):
        """
        Set the default values of the required master tags
        """
        mt = self.master.tags

        # other required tags and their defaults
        mt['[dt_sim]'] = 0.01
        mt['[hawc2_exe]'] = 'hawc2-latest'
        # convergence_limits  0.001  0.005  0.005 ;
        # critical one, risidual on the forces: 0.0001 = 1e-4
        mt['[epsresq]'] = '1.0' # default=10.0
        # increment residual
        mt['[epsresd]'] = '0.5' # default= 1.0
        # constraint equation residual
        mt['[epsresg]'] = '1e-8' # default= 1e-7
        # folder names for the saved results, htc, data, zip files
        # Following dirs are relative to the model_dir_server and they specify
        # the location of where the results, logfiles, animation files that where
        # run on the server should be copied to after the simulation has finished.
        # on the node, it will try to copy the turbulence files from these dirs
        mt['[animation_dir]'] = 'animation/'
        mt['[control_dir]']   = 'control/'
        mt['[data_dir]']      = 'data/'
        mt['[eigen_analysis]'] = False
        mt['[eigenfreq_dir]'] = False
        mt['[htc_dir]']       = 'htc/'
        mt['[log_dir]']       = 'logfiles/'
        mt['[meander_dir]']   = False
        mt['[opt_dir]']       = False
        mt['[pbs_out_dir]']   = 'pbs_out/'
        mt['[res_dir]']       = 'res/'
        mt['[iter_dir]']      = 'iter/'
        mt['[turb_dir]']      = 'turb/'
        mt['[turb_db_dir]']   = '../turb/'
        mt['[wake_dir]']      = False
        mt['[hydro_dir]']     = False
        mt['[mooring_dir]']   = False
        mt['[externalforce]'] = False
        mt['[Case folder]']   = 'NoCaseFolder'
        # zip_root_files only is used when copy to run_dir and zip creation, define
        # in the HtcMaster object
        mt['[zip_root_files]'] = []
        # only active on PBS level, so files have to be present in the run_dir
        mt['[copyback_files]'] = []   # copyback_resultfile
        mt['[copyback_frename]'] = [] # copyback_resultrename
        mt['[copyto_files]'] = []     # copyto_inputfile
        mt['[copyto_generic]'] = []   # copyto_input_required_defaultname

        # In master file tags within the HAWC2 vs HAWCStab2 context
        mt['[hawc2]'] = False
        mt['[output]'] = False
        mt['[eigen_analysis]'] = False
        mt['[system_eigen_analysis]'] = False
        mt['[operational_data]'] = 'case_name.opt'

        mt['[gravity]'] = 0.0
        mt['[shaft_tilt]'] = 0.0 # 5.0
        mt['[coning]'] = 0.0 # 2.5
        mt['[Windspeed]'] = 1.0
        mt['[wtilt]'] = 0.0
        mt['[wdir]'] = 0.0
        mt['[aerocalc]'] = 1
        mt['[Induction]'] = 0
        mt['[tip_loss]'] = 0
        mt['[Dyn stall]'] = 0
        mt['[tu_model]'] = 0
        mt['[shear_exp]'] = 0
        mt['[tower_shadow]'] = 0
        mt['[TI]'] = 1
        mt['[fixed_omega]'] = 1.0
        mt['[init_wr]'] = 0
        mt['[pc_file_name]'] = 'hawc_pc.mhh'
        mt['[ae_file_name]'] = 'hawc2_ae.mhh'
        mt['[nr_ae_sections]'] = 30
        mt['[use_nr_ae_sections]'] = True
        mt['[use_ae_distrb_file]'] = False
        mt['[ae_set_nr]'] = 1
        # tors_e output depends on the pitch axis configuration
        mt['[c12]'] = False
        mt['[c14]'] = False

        mt['[t0]'] = 500
        mt['[time stop]'] = 600

        mt['[hs2]'] = False
        mt['[nr_blade_modes_hs2]'] = 10
        mt['[stab_analysis]'] = False
        mt['[steady_states]'] = True
        mt['[hs2_bladedeform_switch]'] = True
        mt['[hs2_gradients_switch]'] = False
        # by default take the stiff set
        mt['[st_file]'] = 'hawc2_st.mhh'
        mt['[tower_set]'] = 4 # 1
        mt['[shaft_set]'] = 4 # 2
        mt['[blade_set]'] = 4 # 3
        mt['[tower_subset]'] = 1
        mt['[shaft_subset]'] = 1
        mt['[blade_subset]'] = 1
        mt['[blade_nbodies]'] = 1
        mt['[blade_posx]'] = -0.75
        mt['[blade_damp_x]'] = 0.01
        mt['[blade_damp_y]'] = 0.01
        mt['[blade_damp_z]'] = 0.01
        # HAWCStab2 convergence criteria
        mt['[bem_tol]'] = 1e-12
        mt['[bem_itmax]'] = 10000
        mt['[bem_1relax]'] = 0.02
        mt['[ae_tolrel]'] = 1e-7
        mt['[ae_itmax]'] = 2000
        mt['[ae_1relax]'] = 0.5
        mt['[tol_7]'] = 10
        mt['[tol_8]'] = 5
        mt['[tol_9]'] = 1e-8

        # =========================================================================
        # basic required tags by HtcMaster and PBS in order to function properly
        # =========================================================================
        # the express queue ('#PBS -q xpresq') has a maximum walltime of 1h
        mt['[pbs_queue_command]'] = '#PBS -q workq'
        # walltime should have following format: hh:mm:ss
        mt['[walltime]'] = '04:00:00'
        mt['[auto_walltime]'] = False

    def get_dlc_casedefs(self):
        """
        Create iter_dict and opt_tags based on spreadsheets
        """

        iter_dict = dict()
        iter_dict['[empty]'] = [False]

        # see if a htc/DLCs dir exists
        dlcs_dir = os.path.join(self.P_SOURCE, 'htc', 'DLCs')
        if os.path.exists(dlcs_dir):
            opt_tags = dlcdefs.excel_stabcon(dlcs_dir)
        else:
            opt_tags = dlcdefs.excel_stabcon(os.path.join(self.P_SOURCE, 'htc'))

        if len(opt_tags) < 1:
            raise ValueError('There are is not a single case defined. Make sure '
                             'the DLC spreadsheets are configured properly.')

        # add all the root files, except anything with *.zip
        f_ziproot = []
        for (dirpath, dirnames, fnames) in os.walk(self.P_SOURCE):
            # remove all zip files
            for i, fname in enumerate(fnames):
                if fname.endswith('.zip'):
                    fnames.pop(i)
            f_ziproot.extend(fnames)
            break
        # and add those files
        for opt in opt_tags:
            opt['[zip_root_files]'] = f_ziproot

        self.master.output_dirs.extend('[Case folder]')
        self.master.output_dirs.extend('[Case id.]')

        return iter_dict, opt_tags

    def create_inputs(self, iter_dict, opt_tags):

        sim.prepare_launch(iter_dict, opt_tags, self.master, self._var_tag_func,
                           write_htc=True, runmethod=self.runmethod, verbose=False,
                           copyback_turb=False, msg='', update_cases=False,
                           ignore_non_unique=False, run_only_new=False,
                           pbs_fname_appendix=False, short_job_names=False)

    def get_control_tuning(self, fpath):
        """
        Read a HAWCStab2 controller tuning file and return as tags
        """
        tuning = hs2.hs2_control_tuning()
        tuning.read_parameters(fpath)

        tune_tags = {}

        tune_tags['[pi_gen_reg1.K]'] = tuning.pi_gen_reg1.K

        tune_tags['[pi_gen_reg2.I]'] = tuning.pi_gen_reg2.I
        tune_tags['[pi_gen_reg2.Kp]'] = tuning.pi_gen_reg2.Kp
        tune_tags['[pi_gen_reg2.Ki]'] = tuning.pi_gen_reg2.Ki

        tune_tags['[pi_pitch_reg3.Kp]'] = tuning.pi_pitch_reg3.Kp
        tune_tags['[pi_pitch_reg3.Ki]'] = tuning.pi_pitch_reg3.Ki
        tune_tags['[pi_pitch_reg3.K1]'] = tuning.pi_pitch_reg3.K1
        tune_tags['[pi_pitch_reg3.K2]'] = tuning.pi_pitch_reg3.K2

        tune_tags['[aero_damp.Kp2]'] = tuning.aero_damp.Kp2
        tune_tags['[aero_damp.Ko1]'] = tuning.aero_damp.Ko1
        tune_tags['[aero_damp.Ko2]'] = tuning.aero_damp.Ko2

        return tune_tags

    def post_processing(self, statistics=True, resdir=None):
        """
        Parameters
        ----------

        resdir : str, default=None
            Defaults to reading the results from the [run_dir] tag.
            Force to any other directory using this variable. You can also use
            the presets as defined for runmethod in _set_path_config.
        """

        post_dir = self.POST_DIR

        # =========================================================================
        # check logfiles, results files, pbs output files
        # logfile analysis is written to a csv file in logfiles directory
        # =========================================================================
        # load the file saved in post_dir
        cc = sim.Cases(post_dir, self.sim_id, rem_failed=False)

        if resdir is None:
            # we keep the run_dir as defined during launch
            run_root = None
        elif resdir in ['local', 'local-script', 'none', 'local-ram']:
            run_root = '/home/dave/SimResults'
        elif resdir == 'windows-script':
            run_root = '/mnt/D16731/dave/Documents/_SimResults'
        elif resdir == 'gorm':
            run_root = '/mnt/hawc2sim/h2_vs_hs2'
        elif resdir == 'jess':
            run_root = '/mnt/hawc2sim/h2_vs_hs2'
        else:
            run_root = None
            cc.change_results_dir(resdir)

        if isinstance(run_root, str):
            forcedir = os.path.join(run_root, self.PROJECT, self.sim_id)
            cc.change_results_dir(forcedir)

        cc.post_launch()
        cc.remove_failed()

        if statistics:
            tags=['[windspeed]']
            stats_df = cc.statistics(calc_mech_power=False, ch_fatigue=[],
                                     tags=tags, update=False)
            ftarget = os.path.join(self.POST_DIR, '%s_statistics.xlsx')
            stats_df.to_excel(ftarget % self.sim_id)


class MappingsH2HS2(object):

    def __init__(self, chord_length=3.0):
        """
        """
        self.hs2_res = hs2.results()
        self.chord_length = chord_length

    def powercurve(self, h2_df_stats, fname_hs):

        self._powercurve_h2(h2_df_stats)
        self._powercurve_hs2(fname_hs)

    def _powercurve_h2(self, df_stats):

        mappings = {'Ae rot. power' : 'P_aero',
                    'Ae rot. thrust': 'T_aero',
                    'Vrel-1-39.03'  : 'vrel_39',
                    'Omega'         : 'rotorspeed',
                    'tower-tower-node-010-forcevec-y' : 'T_towertop',
                    'tower-shaft-node-003-forcevec-y' : 'T_shafttip'}

        df_stats.sort_values('[windspeed]', inplace=True)
        df_mean = pd.DataFrame()
        df_std = pd.DataFrame()

        for key, value in mappings.items():
            tmp = df_stats[df_stats['channel']==key]
            df_mean[value] = tmp['mean'].values.copy()
            df_std[value] = tmp['std'].values.copy()

        # also add the wind speed
        df_mean['windspeed'] = tmp['[windspeed]'].values.copy()
        df_std['windspeed'] = tmp['[windspeed]'].values.copy()

        self.pwr_h2_mean = df_mean
        self.pwr_h2_std = df_std
        self.h2_df_stats = df_stats

    def _powercurve_hs2(self, fname):

        mappings = {'P [kW]'  :'P_aero',
                    'T [kN]'  :'T_aero',
                    'V [m/s]' :'windspeed'}

        df_pwr, units = self.hs2_res.load_pwr_df(fname)

        self.pwr_hs = pd.DataFrame()
        for key, value in mappings.items():
            self.pwr_hs[value] = df_pwr[key].values.copy()

    def blade_distribution(self, fname_h2, fname_hs2, h2_df_stats=None,
                           fname_h2_tors=None):

        self.hs2_res.load_ind(fname_hs2)
        self.h2_res = sim.windIO.ReadOutputAtTime(fname_h2)
        self._distribution_hs2()
        self._distribution_h2()
        if h2_df_stats is not None:
            self.h2_df_stats = h2_df_stats
            if fname_h2_tors is not None:
                self.distribution_torsion_h2(fname_h2_tors)

    def _distribution_hs2(self):
        """Read a HAWCStab2 *.ind file (blade distribution loading)
        """

        mapping_hs2 =  {'s [m]'       :'curved_s',
                        'CL0 [-]'     :'Cl',
                        'CD0 [-]'     :'Cd',
                        'CT [-]'      :'Ct',
                        'CP [-]'      :'Cp',
                        'A [-]'       :'ax_ind',
                        'AP [-]'      :'tan_ind',
                        'U0 [m/s]'    :'vrel',
                        'PHI0 [rad]'  :'inflow_angle',
                        'ALPHA0 [rad]':'AoA',
                        'X_AC0 [m]'   :'pos_x',
                        'Y_AC0 [m]'   :'pos_y',
                        'Z_AC0 [m]'   :'pos_z',
                        'UX0 [m]'     :'def_x',
                        'UY0 [m]'     :'def_y',
                        'Tors. [rad]' :'torsion',
                        'Twist[rad]'  :'twist',
                        'V_a [m/s]'   :'ax_ind_vel',
                        'V_t [m/s]'   :'tan_ind_vel',
                        'FX0 [N/m]'   :'F_x',
                        'FY0 [N/m]'   :'F_y',
                        'M0 [Nm/m]'   :'M'}

        try:
            hs2_cols = [k for k in mapping_hs2]
            # select only the HS channels that will be used for the mapping
            std_cols = [mapping_hs2[k] for k in hs2_cols]
            self.hs_aero = self.hs2_res.ind.df_data[hs2_cols].copy()
        except KeyError:
            # some results have been created with older HAWCStab2 that did not
            # include CT and CP columns
            mapping_hs2.pop('CT [-]')
            mapping_hs2.pop('CP [-]')
            hs2_cols = [k for k in mapping_hs2]
            std_cols = [mapping_hs2[k] for k in hs2_cols]
            # select only the HS channels that will be used for the mapping
            self.hs_aero = self.hs2_res.ind.df_data[hs2_cols].copy()

        # change column names to the standard form that is shared with H2
        self.hs_aero.columns = std_cols
        self.hs_aero['AoA'] *= (180.0/np.pi)
        self.hs_aero['inflow_angle'] *= (180.0/np.pi)
        self.hs_aero['torsion'] *= (180.0/np.pi)
#        self.hs_aero['pos_x'] = (-1.0) # self.chord_length / 4.0

    def _distribution_h2(self):
        mapping_h2 =  { 'Radius_s'  :'curved_s',
                        'Cl'        :'Cl',
                        'Cd'        :'Cd',
                        'Ct_local'  :'Ct',
                        'Cq_local'  :'Cq',
                        'Induc_RPy' :'ax_ind_vel',
                        'Induc_RPx' :'tan_ind_vel',
                        'Vrel'      :'vrel',
                        'Inflow_ang':'inflow_angle',
                        'alfa'      :'AoA',
                        'pos_RP_x'  :'pos_x',
                        'pos_RP_y'  :'pos_y',
                        'pos_RP_z'  :'pos_z',
                        'Secfrc_RPx':'F_x',
                        'Secfrc_RPy':'F_y',
                        'Secmom_RPz':'M'}
        h2_cols = [k for k in mapping_h2]
        std_cols = [mapping_h2[k] for k in h2_cols]

        # select only the h2 channels that will be used for the mapping
        h2_aero = self.h2_res[h2_cols].copy()
        # change column names to the standard form that is shared with HS
        h2_aero.columns = std_cols
        h2_aero['def_x'] = self.h2_res['Pos_B_x'] - self.h2_res['Inipos_x_x']
        h2_aero['def_y'] = self.h2_res['Pos_B_y'] - self.h2_res['Inipos_y_y']
        h2_aero['def_z'] = self.h2_res['Pos_B_z'] - self.h2_res['Inipos_z_z']
        h2_aero['ax_ind_vel'] *= (-1.0)
        h2_aero['pos_x'] += (self.chord_length / 2.0)
        h2_aero['F_x'] *= (1e3)
        h2_aero['F_y'] *= (1e3)
        h2_aero['M'] *= (1e3)
        h2_aero['M'] -= (h2_aero['F_y']*1.5)
#        # HAWC2 includes root and tip nodes, while HAWC2 doesn't. Remove them
#        h2_aero = h2_aero[1:-1]
        self.h2_aero = h2_aero

    def distribution_torsion_h2(self, fname_h2):
        """Determine torsion distribution from the HAWC2 result statistics.
        tors_e is in degrees.
        """
        if not hasattr(self, 'h2_aero'):
            raise UserWarning('first run blade_distribution')

        # load the HAWC2 .sel file for the channels
        fpath = os.path.dirname(fname_h2)
        fname = os.path.basename(fname_h2)
        res = sim.windIO.LoadResults(fpath, fname, readdata=False)
        sel = res.ch_df[res.ch_df.sensortype == 'Tors_e'].copy()
        sel.sort_values(['radius'], inplace=True)
        self.h2_aero['Radius_s_tors'] = sel.radius.values.copy()
        self.h2_aero['tors_e'] = sel.radius.values.copy()
        tors_e_channels = sel.ch_name.tolist()

        # find the current case in the statistics DataFrame
        case = fname.replace('.htc', '')
        df_case = self.h2_df_stats[self.h2_df_stats['[case_id]']==case].copy()
        # and select all the torsion channels
        df_tors_e = df_case[df_case.channel.isin(tors_e_channels)].copy()
        # join the stats with the channel descriptions DataFrames, have the
        # same name on the joining column
        df_tors_e.set_index('channel', inplace=True)
        sel.set_index('ch_name', inplace=True)

        # joining happens on the index, and for which the same channel has been
        # used: the unique HAWC2 channel naming scheme
        df_tors_e = pd.concat([df_tors_e, sel], axis=1)
        df_tors_e.radius = df_tors_e.radius.astype(np.float64)
        # sorting on radius, combine with ch_df
        df_tors_e.sort_values(['radius'], inplace=True)

        # FIXME: what if number of torsion outputs is less than aero
        # calculation points?
#        df_tmp = pd.DataFrame()
        self.h2_aero['torsion'] = df_tors_e['mean'].values.copy()
        self.h2_aero['torsion_std'] = df_tors_e['std'].values.copy()
        self.h2_aero['torsion_radius_s'] = df_tors_e['radius'].values.copy()
#        df_tmp = pd.DataFrame()
#        df_tmp['torsion'] = df_tors_e['mean'].copy()
#        df_tmp['torsion_std'] = df_tors_e['std'].copy()
#        df_tmp['torsion_radius_s'] = df_tors_e['radius'].copy()
#        df_tmp.set_index('')

    def body_structure_modes(self, fname_h2, fname_hs):
        self._body_structure_modes_h2(fname_h2)
        self._body_structure_modes_hs(fname_hs)

    def _body_structure_modes_h2(self, fname):
        self.body_freq_h2 = sim.windIO.ReadEigenBody(fname)

        blade_h2 = self.body_freq_h2[self.body_freq_h2['body']=='blade1'].copy()
        # because HAWCStab2 is sorted by frequency
        blade_h2.sort_values('Fd_hz', inplace=True)
        # HAWC2 usually has a lot of duplicate entries
        blade_h2.drop_duplicates('Fd_hz', keep='first', inplace=True)
        # also drop the ones with very high damping, and 0 frequency
        query = '(log_decr_pct < 500 and log_decr_pct > -500) and Fd_hz > 0.0'
        self.blade_body_freq_h2 = blade_h2.query(query)

    def _body_structure_modes_hs(self, fname):
        self.body_freq_hs = hs2.results().load_cmb_df(fname)


class Plots(object):
    """
    Comparison plots between HACW2 and HAWCStab2. This is done based on
    the HAWC2 output output_at_time, and HAWCStab2 output *.ind
    """

    def __init__(self):

        self.h2c = 'b'
        self.h2ms = '+'
        self.h2ls = '-'
        self.hsc = 'r'
        self.hsms = 'x'
        self.hsls = '--'
        self.errls = '-'
        self.errc = 'k'
        self.errms = 'x'
#        self.errlab = 'diff [\\%]'
        self.errlab = 'diff'
        self.interactive = False

        self.dist_size = (16, 11)
        self.dist_nrows = 3
        self.dist_ncols = 4
        self.dist_channels = ['pos_x', 'pos_y', 'AoA', 'inflow_angle',
                              'Cl', 'Cd', 'vrel', 'ax_ind_vel',
                              'F_x', 'F_y', 'M', 'torsion']

    def load_h2(self, fname_h2, h2_df_stats=None, fname_h2_tors=None):

        res = MappingsH2HS2()
        res.h2_res = sim.windIO.ReadOutputAtTime(fname_h2)
        res._distribution_h2()
        if h2_df_stats is not None:
            res.h2_df_stats = h2_df_stats
            if fname_h2_tors is not None:
                res.distribution_torsion_h2(fname_h2_tors)

        return res

    def load_hs(self, fname_hs):

        res = MappingsH2HS2()
        res.hs2_res.load_ind(fname_hs)
        res._distribution_hs2()

        return res

    def new_fig(self, title=None, nrows=2, ncols=1, dpi=150, size=(12.0, 5.0)):

        if self.interactive:
            subplots = plt.subplots
        else:
            subplots = mplutils.subplots

        fig, axes = subplots(nrows=nrows, ncols=ncols, dpi=dpi, figsize=size)
        axes = axes.ravel()
        if title is not None:
            fig.suptitle(title)
        return fig, axes

    def set_axes_label_grid(self, axes, setlegend=False):
        for ax in axes.ravel():
            if setlegend:
                leg = ax.legend(loc='best')
                if leg is not None:
                    leg.get_frame().set_alpha(0.5)
            ax.grid(True)
        return axes

    def save_fig(self, fig, axes, fname):
        fig.tight_layout()
        fig.subplots_adjust(top=0.89)
        fig.savefig(fname, dpi=150)
        fig.clear()
        print('saved:', fname)

    def distribution(self, results, labels, title, channels, x_ax='pos_z',
                     xlabel='Z-coordinate [m]', nrows=2, ncols=4, size=(16, 5),
                     i0=1):
        """
        Compare blade distribution results
        """
        res1 = results[0]
        res2 = results[1]
        lab1 = labels[0]
        lab2 = labels[1]

        radius1 = res1[x_ax].values
        radius2 = res2[x_ax].values

        fig, axes = self.new_fig(title=title, nrows=nrows, ncols=ncols, size=size)
        axesflat = axes.flatten()
        for i, chan in enumerate(channels):
            ax = axesflat[i]
            ax.plot(radius1, res1[chan].values, color=self.h2c,
                    label=lab1, alpha=0.9, marker=self.h2ms, ls=self.h2ls)
            ax.plot(radius2, res2[chan].values, color=self.hsc,
                    label=lab2, alpha=0.7, marker=self.hsms, ls=self.hsls)
            ax.set_ylabel(chan.replace('_', '\\_'))
            xlim = max(radius1.max(), radius2.max())
            ax.set_xlim([0, xlim])

#            if len(radius1) > len(radius2):
#                radius = res1.hs_aero['pos_z'].values[n0:]
#                x = res2.hs_aero['pos_z'].values[n0:]
#                y = res2.hs_aero[chan].values[n0:]
#                qq1 = res1.hs_aero[chan].values[n0:]
#                qq2 = interpolate.griddata(x, y, radius)
#            elif len(radius1) < len(radius2):
#                radius = res2.hs_aero['pos_z'].values[n0:]
#                x = res1.hs_aero['pos_z'].values[n0:]
#                y = res1.hs_aero[chan].values[n0:]
#                qq1 = interpolate.griddata(x, y, radius)
#                qq2 = res2.hs_aero[chan].values[n0:]
#            else:
#                if np.allclose(radius1, radius2):
#                    radius = res1.hs_aero['pos_z'].values[n0:]
#                    qq1 = res1.hs_aero[chan].values[n0:]
#                    qq2 = res2.hs_aero[chan].values[n0:]
#                else:
#                    radius = res1.hs_aero['pos_z'].values[n0:]
#                    x = res2.hs_aero['pos_z'].values[n0:]
#                    y = res2.hs_aero[chan].values[n0:]
#                    qq1 = res1.hs_aero[chan].values[n0:]
#                    qq2 = interpolate.griddata(x, y, radius)

            # relative errors on the right axes
#            err = np.abs(1.0 - (res1[chan].values / res2[chan].values))*100.0
            # absolute errors on the right axes
            err = res1[chan].values[i0:] - res2[chan].values[i0:]
            axr = ax.twinx()
            axr.plot(radius1[i0:], err, color=self.errc, ls=self.errls,
                     alpha=0.6, label=self.errlab, marker=self.errms)
#            if err.max() > 50:
#                axr.set_ylim([0, 35])

            # use axr for the legend, but only for the first plot
            if i == 0:
                lines = ax.lines + axr.lines
                labels = [l.get_label() for l in lines]
                leg = axr.legend(lines, labels, loc='best')
                leg.get_frame().set_alpha(0.5)

        # x-label only on the last row
        for k in range(ncols):
            axesflat[-k-1].set_xlabel(xlabel)

        axes = self.set_axes_label_grid(axes)
        return fig, axes

    def all_h2_channels(self, results, labels, fpath, channels=None):
        """Results is a list of res (=HAWC2 results object)"""

        for chan, details in results[0].ch_dict.items():
            if channels is None or chan not in channels:
                continue
            resp = []
            for res in results:
                resp.append([res.sig[:,0], res.sig[:,details['chi']]])

            fig, axes = self.new_fig(title=chan.replace('_', '\\_'))
            try:
                mplutils.time_psd(resp, labels, axes, alphas=[1.0, 0.7], NFFT=None,
                                   colors=['k-', 'r-'], res_param=250, f0=0, f1=5,
                                   nr_peaks=10, min_h=15, mark_peaks=False)
            except Exception as e:
                print('****** FAILED')
                print(e)
                continue
            axes[0].set_xlim([0,5])
            axes[1].set_xlim(res.sig[[0,-1],0])
            fname = os.path.join(fpath, chan + '.png')
            self.save_fig(fig, axes, fname)

    def h2_blade_distribution(self, fname_1, fname_2, title, labels, n0=0,
                              df_stats1=None, df_stats2=None):
        """
        Compare blade distribution aerodynamics of two HAWC2 cases.
        """
        tors1 = fname_1.split('_aero_at_tstop')[0]
        res1 = self.load_h2(fname_1, h2_df_stats=df_stats1, fname_h2_tors=tors1)
        tors2 = fname_2.split('_aero_at_tstop')[0]
        res2 = self.load_h2(fname_2, h2_df_stats=df_stats2, fname_h2_tors=tors2)

        results = [res1.h2_aero[n0+1:], res2.h2_aero[n0+1:]]

        fig, axes = self.distribution(results, labels, title, self.dist_channels,
                                      x_ax='pos_z', xlabel='Z-coordinate [m]',
                                      nrows=self.dist_nrows,
                                      ncols=self.dist_ncols,
                                      size=self.dist_size)

        return fig, axes

    def hs_blade_distribution(self, fname_1, fname_2, title, labels, n0=0):

        res1 = self.load_hs(fname_1)
        res2 = self.load_hs(fname_2)

        results = [res1.hs_aero[n0:], res2.hs_aero[n0:]]
#        channels = ['pos_x', 'pos_y', 'AoA', 'inflow_angle', 'Cl', 'Cd',
#                    'vrel', 'ax_ind_vel']

        fig, axes = self.distribution(results, labels, title, self.dist_channels,
                                      x_ax='pos_z', xlabel='Z-coordinate [m]',
                                      nrows=self.dist_nrows,
                                      ncols=self.dist_ncols,
                                      size=self.dist_size)

        return fig, axes

    def blade_distribution(self, fname_h2, fname_hs2, title, n0=0,
                           h2_df_stats=None, fname_h2_tors=None):
        """Compare aerodynamics, blade deflections between HAWC2 and HAWCStab2.
        This is based on HAWCSTab2 *.ind files, and an HAWC2 output_at_time
        output file.
        """

        results = MappingsH2HS2()
        results.blade_distribution(fname_h2, fname_hs2, h2_df_stats=h2_df_stats,
                                   fname_h2_tors=fname_h2_tors)
        res = [results.h2_aero[n0+1:-1], results.hs_aero[n0:]]

#        channels = ['pos_x', 'pos_y', 'AoA', 'inflow_angle', 'Cl', 'Cd',
#                    'vrel', 'ax_ind_vel']
        labels = ['HAWC2', 'HAWCStab2']

        fig, axes = self.distribution(res, labels, title, self.dist_channels,
                                      x_ax='pos_z', xlabel='Z-coordinate [m]',
                                      nrows=self.dist_nrows,
                                      ncols=self.dist_ncols,
                                      size=self.dist_size)

        return fig, axes

    def blade_distribution2(self, fname_h2, fname_hs2, title, n0=0):
        """Compare aerodynamics, blade deflections between HAWC2 and HAWCStab2.
        This is based on HAWCSTab2 *.ind files, and an HAWC2 output_at_time
        output file.
        """

        results = MappingsH2HS2()
        results.blade_distribution(fname_h2, fname_hs2)
        res = [results.h2_aero[n0+1:-1], results.hs_aero[n0:]]

        channels = ['pos_x', 'pos_y', 'torsion', 'inflow_angle',
                    'Cl', 'Cd', 'vrel',  'AoA',
                    'F_x', 'F_y', 'M', 'ax_ind_vel', 'torsion']
        labels = ['HAWC2', 'HAWCStab2']

        fig, axes = self.distribution(res, labels, title, channels,
                                      x_ax='pos_z', xlabel='Z-coordinate [m]',
                                      nrows=3, ncols=4, size=(16, 12))

        return fig, axes

    def powercurve(self, h2_df_stats, fname_hs, title, size=(8.6, 4)):

        results = MappingsH2HS2()
        results.powercurve(h2_df_stats, fname_hs)

        fig, axes = self.new_fig(title=title, nrows=1, ncols=2, size=size)

        wind_h2 = results.pwr_h2_mean['windspeed'].values
        wind_hs = results.pwr_hs['windspeed'].values

        # POWER
        ax = axes[0]
        key = 'P_aero'
        # HAWC2
        yerr = results.pwr_h2_std[key].values
        ax.errorbar(wind_h2, results.pwr_h2_mean[key].values, color=self.h2c,
                    yerr=yerr, marker=self.h2ms, ls=self.h2ls, label='HAWC2',
                    alpha=0.9)
        # HAWCSTAB2
        ax.plot(wind_hs, results.pwr_hs[key].values, label='HAWCStab2',
                alpha=0.7, color=self.hsc, ls=self.hsls, marker=self.hsms)
        ax.set_title('Power [kW]')
        # relative errors on the right axes
        axr = ax.twinx()
        assert np.allclose(wind_h2, wind_hs)
        qq1 = results.pwr_h2_mean[key].values
        qq2 = results.pwr_hs[key].values
        err = np.abs(1.0 - qq1 / qq2)*100.0
        axr.plot(wind_hs, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)
#        axr.set_ylabel('absolute error []')
#        axr.set_ylim([])

        # THRUST
        ax = axes[1]
        keys = ['T_aero', 'T_shafttip']
        lss = [self.h2ls, '--', ':']
        # HAWC2
        for key, ls in zip(keys, lss):
            label = 'HAWC2 %s' % (key.replace('_', '$_{') + '}$')
            yerr = results.pwr_h2_std[key].values
            c = self.h2c
#            print(label)
            ax.errorbar(wind_h2, results.pwr_h2_mean[key].values, color=c, ls=ls,
                        label=label, alpha=0.9, yerr=yerr, marker=self.h2ms)
        # HAWCStab2
        ax.plot(wind_hs, results.pwr_hs['T_aero'].values, color=self.hsc, alpha=0.7,
                label='HAWCStab2 T$_{aero}$', marker=self.hsms, ls=self.hsls)
        # relative errors on the right axes
        axr = ax.twinx()
        qq1 = results.pwr_h2_mean['T_aero'].values
        qq2 = results.pwr_hs['T_aero'].values
        err = np.abs(1.0 - (qq1 / qq2))*100.0
        axr.plot(wind_hs, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)
        ax.set_title('Thrust [kN]')

        axes = self.set_axes_label_grid(axes, setlegend=True)
#        # use axr for the legend
#        keys_tex = [kk.replace('_', '$_{') + '}$' for kk in keys]
#        lines = [ax.lines[2]] + [ax.lines[5]] + [ax.lines[6]] + axr.lines
#        labels = keys_tex + ['HAWCStab2 T$_{aero}$', self.errlab]
#        leg = axr.legend(lines, labels, loc='best')
#        leg.get_frame().set_alpha(0.5)

        return fig, axes

    def h2_powercurve(self, h2_df_stats1, h2_df_stats2, title, labels,
                      size=(8.6,4)):
        res1 = MappingsH2HS2()
        res1._powercurve_h2(h2_df_stats1)
        wind1 = res1.pwr_h2_mean['windspeed'].values

        res2 = MappingsH2HS2()
        res2._powercurve_h2(h2_df_stats2)
        wind2 = res2.pwr_h2_mean['windspeed'].values

        fig, axes = self.new_fig(title=title, nrows=1, ncols=2, size=size)

        # POWER
        ax = axes[0]
        key = 'P_aero'
        # HAWC2
        yerr1 = res1.pwr_h2_std[key].values
        ax.errorbar(wind1, res1.pwr_h2_mean[key].values, color=self.h2c, yerr=yerr1,
                    marker=self.h2ms, ls=self.h2ls, label=labels[0], alpha=0.9)
        yerr2 = res2.pwr_h2_std[key]
        ax.errorbar(wind2, res2.pwr_h2_mean[key].values, color=self.hsc, yerr=yerr2,
                    marker=self.hsms, ls=self.hsls, label=labels[1], alpha=0.7)
        ax.set_title('Power [kW]')
        # relative errors on the right axes
        axr = ax.twinx()
        assert np.allclose(wind1, wind2)
        qq1 = res1.pwr_h2_mean[key].values
        qq2 = res2.pwr_h2_mean[key].values
        err = np.abs(1.0 - qq1 / qq2)*100.0
        axr.plot(wind1, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)

        # THRUST
        ax = axes[1]
        keys = ['T_aero', 'T_shafttip']
        lss = [self.h2ls, '--', ':']
        for key, ls in zip(keys, lss):
            label = '%s %s' % (labels[0], key.replace('_', '$_{') + '}$')
            yerr = res1.pwr_h2_std[key].values
            c = self.h2c
            ax.errorbar(wind1, res1.pwr_h2_mean[key].values, color=c, ls=ls,
                        label=label, alpha=0.9, yerr=yerr, marker=self.h2ms)
        for key, ls in zip(keys, lss):
            label = '%s %s' % (labels[1], key.replace('_', '$_{') + '}$')
            yerr = res2.pwr_h2_std[key].values
            c = self.hsc
            ax.errorbar(wind2, res2.pwr_h2_mean[key].values, color=c, ls=ls,
                        label=label, alpha=0.9, yerr=yerr, marker=self.hsms)
        # relative errors on the right axes
        axr = ax.twinx()
        qq1 = res1.pwr_h2_mean['T_aero'].values
        qq2 = res2.pwr_h2_mean['T_aero'].values
        err = np.abs(1.0 - (qq1 / qq2))*100.0
        axr.plot(wind1, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)
        ax.set_title('Thrust [kN]')

        axes = self.set_axes_label_grid(axes, setlegend=True)
#        # use axr for the legend
#        lines = ax.lines + axr.lines
#        labels = [l.get_label() for l in lines]
#        leg = axr.legend(lines, labels, loc='best')
#        leg.get_frame().set_alpha(0.5)

        return fig, axes

    def hs_powercurve(self, fname1, fname2, title, labels, size=(8.6, 4)):

        res1 = MappingsH2HS2()
        res1._powercurve_hs2(fname1)
        wind1 = res1.pwr_hs['windspeed'].values

        res2 = MappingsH2HS2()
        res2._powercurve_hs2(fname2)
        wind2 = res2.pwr_hs['windspeed'].values

        fig, axes = self.new_fig(title=title, nrows=1, ncols=2, size=size)

        # POWER
        ax = axes[0]
        key = 'P_aero'
        ax.plot(wind1, res1.pwr_hs['P_aero'].values, label=labels[0],
                alpha=0.9, color=self.h2c, ls=self.h2ls, marker=self.h2ms)
        ax.plot(wind2, res2.pwr_hs['P_aero'].values, label=labels[1],
                alpha=0.7, color=self.hsc, ls=self.hsls, marker=self.hsms)
        ax.set_title('Power [kW]')
        # relative errors on the right axes
        axr = ax.twinx()
        assert np.allclose(wind1, wind2)
        qq1 = res1.pwr_hs[key].values
        qq2 = res2.pwr_hs[key].values
        err = np.abs(1.0 - qq1 / qq2)*100.0
        axr.plot(wind1, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)
#        axr.set_ylim([])

        # THRUST
        ax = axes[1]
        ax.plot(wind1, res1.pwr_hs['T_aero'].values, color=self.h2c, alpha=0.9,
                label=labels[0], marker=self.h2ms, ls=self.h2ls)
        ax.plot(wind2, res2.pwr_hs['T_aero'].values, color=self.hsc, alpha=0.7,
                label=labels[1], marker=self.hsms, ls=self.hsls)
        # relative errors on the right axes
        axr = ax.twinx()
        qq1 = res1.pwr_hs['T_aero'].values
        qq2 = res2.pwr_hs['T_aero'].values
        err = np.abs(1.0 - (qq1 / qq2))*100.0
        axr.plot(wind1, err, color=self.errc, ls=self.errls, alpha=0.6,
                 label=self.errlab)
        ax.set_title('Thrust [kN]')

        axes = self.set_axes_label_grid(axes, setlegend=True)
#        # use axr for the legend
#        lines = ax.lines + axr.lines
#        labels = [l.get_label() for l in lines]
#        leg = axr.legend(lines, labels, loc='best')
#        leg.get_frame().set_alpha(0.5)

        return fig, axes


if __name__ == '__main__':

    dummy = None
