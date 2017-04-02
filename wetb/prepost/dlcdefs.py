# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:01:25 2014

@author: dave
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import str
from future.utils import viewitems
from future import standard_library
standard_library.install_aliases()

import os
import unittest
from glob import glob

import pandas as pd

from wetb.prepost import misc

def casedict2xlsx():
    """
    Convert a full Cases.cases dict to Excel spreadsheets
    """


def configure_dirs(verbose=False):
    """
    Automatically configure required directories to launch simulations
    """

    P_RUN = str(os.getcwd())
    p_run_root = os.sep.join(P_RUN.split(os.sep)[:-2])
    # MODEL SOURCES, exchanche file sources
    P_SOURCE = P_RUN
    # Project name, sim_id: derive from folder name
    PROJECT = P_RUN.split(os.sep)[-2]
    sim_id = P_RUN.split(os.sep)[-1]

    master = find_master_file(P_SOURCE)
    if master is None:
        raise ValueError('Could not find master file in htc/_master')
    MASTERFILE = master
    P_MASTERFILE = os.path.join(P_SOURCE, 'htc%s_master%s' % (os.sep, os.sep))
    POST_DIR = os.path.join(p_run_root, PROJECT, sim_id, 'prepost-data%s' % os.sep)

    if verbose:
        print('='*79)
        print('POST_DIR: %s' % POST_DIR)
        print('   P_RUN: %s' % P_RUN)
        print('P_SOURCE: %s' % P_SOURCE)
        print(' PROJECT: %s' % PROJECT)
        print('  sim_id: %s' % sim_id)
        print('  master: %s' % MASTERFILE)
        print('='*79)

    return P_RUN, P_SOURCE, PROJECT, sim_id, P_MASTERFILE, MASTERFILE, POST_DIR


def find_master_file(proot, htc_dir='htc', master_dir='_master',
                     pattern='*_master_*'):
    """
    Find the master file name. It is assumed that the master file is in the
    folder _master, under htc, and contains _master_ in the file name. If
    multiple files contain pattern, the last file of the sorted list is
    returned.

    Parameters
    ----------

    proot

    htc_dir : str, default: htc

    master_dir : str, default: _master

    pattern : str, default: *_master_*

    """

    fpath_search = os.path.join(proot, htc_dir, master_dir, pattern)
    files = glob(fpath_search)
    if len(files) > 0:
        return sorted(files)[-1]
    return None


def variable_tag_func(master, case_id_short=False):
    """
    When using the Excel definitions, and the whole default setup, the
    variable_tag_func is not required to do anything extra.
    """

    # -------------------------------------------------------------------------
#    mt = master
#    V = mt['windspeed']
#    mt['duration'] = mt['time_stop'] - mt['t0']
#    t = mt['duration']
#    if V > abs(1e-15):
#        b = 5.6
#        mt['TI'] = mt['TI_ref'] * ((0.75*V) + b) / V # NTM
#        # ETM
#        c = 2.0
#        V_ave = 0.2 * 50.0
#        sigma = mt['TI_ref'] / V
#        mt['TI'] = sigma * c * (0.072 * (V_ave / c + 3.0) * (V / c - 4.0) + 10.0)
#    else:
#        mt['TI'] = 0
#
#    mt['turb_dx'] = V*t/mt['turb_grid_x']
#
#    mt['turb_dy'] = (mt['rotor_diameter'] / mt['turb_grid_yz'])*1.1
#
#    mt['turb_dz'] = (mt['rotor_diameter'] / mt['turb_grid_yz'])*1.1
#
#    # check: dx spacing should be 0.1*mean_windspeed and 0.2*mean_windspeed
#    # between 0.1 and 0.2 seconds between points
#    if not (V*0.1 < mt['turb_dx'] < V*0.2):
#        logging.warn('turbulence spacing dx out of bounds')
#        print('%5.3f  %5.3f  %5.3f' % (V*0.1, mt['turb_dx'], V*0.2))
#
#    #mt['turb_base_name'] = 'turb_s' + str(mt['turb_seed']) + '_' + str(V)
#    mt['turb_base_name'] = 'turb_s%i_%1.2f' % (mt['turb_seed'], V)
    # -------------------------------------------------------------------------

    return master


def vartags_dlcs(master):

    mt = master.tags

    dlc_case = mt['[Case folder]']
    mt['[data_dir]'] = 'data/'
    mt['[res_dir]'] = 'res/%s/' % dlc_case
    mt['[log_dir]'] = 'logfiles/%s/' % dlc_case
    mt['[htc_dir]'] = 'htc/%s/' % dlc_case
    mt['[case_id]'] = mt['[Case id.]']
    mt['[time_stop]'] = mt['[time stop]']
    mt['[turb_base_name]'] = mt['[Turb base name]']
    mt['[DLC]'] = mt['[Case id.]'].split('_')[0][3:]
    mt['[pbs_out_dir]'] = 'pbs_out/%s/' % dlc_case
    mt['[pbs_in_dir]'] = 'pbs_in/%s/' % dlc_case
    mt['[iter_dir]'] = 'iter/%s/' % dlc_case
    if mt['[eigen_analysis]']:
        rpl = (dlc_case, mt['[Case id.]'])
        mt['[eigenfreq_dir]'] = 'res_eigen/%s/%s/' % rpl
    mt['[duration]'] = str(float(mt['[time_stop]']) - float(mt['[t0]']))
    # replace nan with empty
    for ii, jj in mt.items():
        if jj == 'nan':
            mt[ii] = ''

    return master


def tags_dlcs(master):
    """
    Initiate tags that are defined in the DLC spreadsheets
    """

    master.tags['[t0]'] = 0
    master.tags['[time stop]'] = 0
    master.tags['[Case folder]'] = 'test'
    master.tags['[Case id.]'] = 'test'
    master.tags['[Windspeed]'] = 8
    master.tags['[wdir]'] = 0 # used for the user defined wind
    master.tags['[wdir_rot]'] = 0 # used for the windfield rotations
    master.tags['[tu_seed]'] = 0
    master.tags['[tu_model]'] = 0
    master.tags['[TI]'] = 0
    master.tags['[Turb base name]'] = 'none'
    master.tags['[turb_dx]'] = 1.0
    master.tags['[shear_exp]'] = 0.2
    master.tags['[wsp factor]'] = 1.0
    master.tags['[gust]'] = False
    master.tags['[gust_type]'] = ''
    master.tags['[G_A]'] = ''
    master.tags['[G_phi0]'] = ''
    master.tags['[G_t0]'] = ''
    master.tags['[G_T]'] = ''
    master.tags['[Rotor azimuth]'] = 0
    master.tags['[Free shaft rot]'] = ''
    master.tags['[init_wr]'] = 0.5
    master.tags['[Pitch 1 DLC22b]'] = 0
    master.tags['[Rotor locked]'] = False
    master.tags['[Time stuck DLC22b]'] = -1
    master.tags['[Cut-in time]'] = -1
    master.tags['[Cut-out time]'] = -1
    master.tags['[Stop type]'] = -1
    master.tags['[Pitvel 1]'] = 4
    master.tags['[Pitvel 2]'] = 6
    master.tags['[Grid loss time]'] = 1000
    master.tags['[out_format]'] = 'hawc_binary'
    master.tags['[Time pitch runaway]'] = 1000
    master.tags['[Induction]'] = 1
    master.tags['[Dyn stall]'] = 1
    # required tags for the MannTurb64 standalone turbulence box generator
    master.tags['[MannAlfaEpsilon]'] = 1.0
    master.tags['[MannL]'] = 29.4
    master.tags['[MannGamma]'] = 3.0
    master.tags['[turb_nr_u]'] = 8192
    master.tags['[turb_nr_v]'] = 32
    master.tags['[turb_nr_w]'] = 32
    master.tags['[turb_dx]'] = 1
    master.tags['[turb_dy]'] = 6.5
    master.tags['[turb_dz]'] = 6.5
    master.tags['[high_freq_comp]'] = 1

    return master


def tags_defaults(master):

    # other required tags and their defaults
    master.tags['[dt_sim]'] = 0.02
    master.tags['[hawc2_exe]'] = 'hawc2-latest'
    # folder names for the saved results, htc, data, zip files
    # Following dirs are relative to the model_dir_server and they specify
    # the location of where the results, logfiles, animation files that where
    # run on the server should be copied to after the simulation has finished.
    # on the node, it will try to copy the turbulence files from these dirs
    master.tags['[animation_dir]'] = 'animation/'
    master.tags['[control_dir]']   = 'control/'
    master.tags['[data_dir]']      = 'data/'
    master.tags['[eigen_analysis]'] = False
    master.tags['[eigenfreq_dir]'] = False
    master.tags['[htc_dir]']       = 'htc/'
    master.tags['[log_dir]']       = 'logfiles/'
    master.tags['[meander_dir]']   = False
    master.tags['[opt_dir]']       = False
    master.tags['[pbs_in_dir]']    = 'pbs_in/'
    master.tags['[pbs_out_dir]']   = 'pbs_out/'
    master.tags['[res_dir]']       = 'res/'
    master.tags['[iter_dir]']      = 'iter/'
    master.tags['[turb_dir]']      = 'turb/'
    master.tags['[turb_db_dir]']   = '../turb/'
    master.tags['[wake_dir]']      = False
    master.tags['[hydro_dir]']     = False
    master.tags['[mooring_dir]']   = False
    master.tags['[externalforce]'] = False
    # required tags for the MannTurb64 standalone turbulence box generator
    master.tags['[MannAlfaEpsilon]'] = 1.0
    master.tags['[MannL]'] = 29.4
    master.tags['[MannGamma]'] = 3.0
    master.tags['[turb_nr_u]'] = 8192
    master.tags['[turb_nr_v]'] = 32
    master.tags['[turb_nr_w]'] = 32
    master.tags['[turb_dx]'] = 1
    master.tags['[turb_dy]'] = 6.5
    master.tags['[turb_dz]'] = 6.5
    master.tags['[high_freq_comp]'] = 1
    # zip_root_files only is used when copy to run_dir and zip creation, define
    # in the HtcMaster object
    master.tags['[zip_root_files]'] = []
    # only active on PBS level, so files have to be present in the run_dir
    master.tags['[copyback_files]'] = []   # copyback_resultfile
    master.tags['[copyback_frename]'] = [] # copyback_resultrename
    master.tags['[copyto_files]'] = []     # copyto_inputfile
    master.tags['[copyto_generic]'] = []   # copyto_input_required_defaultname
    master.tags['[eigen_analysis]'] = False

    # =========================================================================
    # basic required tags by HtcMaster and PBS in order to function properly
    # =========================================================================
    # the express queue ('#PBS -q xpresq') has a maximum walltime of 1h
    master.tags['[pbs_queue_command]'] = '#PBS -q workq'
    # walltime should have following format: hh:mm:ss
    master.tags['[walltime]'] = '04:00:00'
    master.tags['[auto_walltime]'] = False

    return master


def excel_stabcon(proot, fext='xlsx', pignore=None, pinclude=None, sheet=0,
                  silent=False):
    """
    Read all MS Excel files that hold load case definitions according to
    the team STABCON definitions. Save each case in a list according to the
    opt_tags principles as used in Simulations.launch(). This method assumes
    that a standard HAWC2 folder layout is used with the following folder
    names: res, logfiles, htc, pbs_out, pbs_in, iter. Further some tags
    are added to be compatible with the tag convention in the Simulations
    module.

    The opt_tags case list is sorted according to the Excel file names, and
    follows the same ordering as in each of the different Excel files.


    Parameters
    ----------

    proot : string
        Path that will be searched recursively for Excel files containing
        load case definitions.

    fext : string, default='xlsx'
        File extension of the Excel files that should be loaded

    pignore : string, default=None
        Specify which string can not occur in the full path of the DLC target.

    pinclude : string, default=None
        Specify which string has to occur in the full path of the DLC target.

    sheet : string or int, default=0
        Name or index of the Excel sheet to be considered. By default, the
        first sheet (index=0) is taken.

    Returns
    -------

    opt_tags : list of dicts
        A list of case dictionaries, where each case dictionary holds all
        the tag/value key pairs for a single given case.

    """
    if not silent:
        print('looking for DLC spreadsheet definitions at:')
        print(proot)
    dict_dfs = misc.read_excel_files(proot, fext=fext, pignore=pignore,
                                    sheet=sheet, pinclude=pinclude,
                                    silent=silent)

    if not silent:
        print('found %i Excel file(s), ' % len(dict_dfs), end='')
    k = 0
    for df in dict_dfs:
        k += len(df)
    if not silent:
        print('in which a total of %s cases are defined.' % k)

    opt_tags = []

    for (dlc, df) in sorted(viewitems(dict_dfs)):
        # replace ';' with False, and Nan(='') with True
        # this is more easy when testing for the presence of stuff compared
        # to checking if a value is either True/False or ''/';'
        # this doesn't work, it will result in 1 for True and 0 for False
        # because the nan values have np.float dtype
#        df.fillna(' ', inplace=True)
#        df.replace(';', False, inplace=True)
        # instead, convert everything to strings, this will maintain some nans
        # as empty strings, but not all of them!
        df2 = df.astype(str)
        for count, row in df2.iterrows():
            tags_dict = {}
            # construct to dict, convert unicode keys/values to strings
            for key, value in row.iteritems():
                if isinstance(value, str):
                    tags_dict[str(key)] = str(value)
                else:
                    tags_dict[str(key)] = value
                # convert ; and empty to False/True
                if isinstance(tags_dict[str(key)], str):
                    if tags_dict[str(key)] == ';':
                        tags_dict[str(key)] = False
                    elif tags_dict[str(key)] == '':
                        tags_dict[str(key)] = True
                    elif tags_dict[str(key)].lower() == 'nan':
                        tags_dict[str(key)] = True

            if '[Windspeed]' not in tags_dict and '[wsp]' in tags_dict:
                tags_dict['[Windspeed]'] = tags_dict['[wsp]']

            tags_dict['[Case folder]'] = tags_dict['[Case folder]'].lower()
            tags_dict['[Case id.]'] = tags_dict['[Case id.]'].lower()
            dlc_case = tags_dict['[Case folder]']
            tags_dict['[data_dir]'] = 'data/'
            tags_dict['[res_dir]'] = 'res/%s/' % dlc_case
            tags_dict['[log_dir]'] = 'logfiles/%s/' % dlc_case
            tags_dict['[htc_dir]'] = 'htc/%s/' % dlc_case
            if '[Case id.]' in tags_dict.keys():
                tags_dict['[case_id]'] = tags_dict['[Case id.]']
            if '[time stop]' in tags_dict.keys():
                tags_dict['[time_stop]'] = tags_dict['[time stop]']
            else:
                tags_dict['[time stop]'] = tags_dict['[time_stop]']
            try:
                tags_dict['[turb_base_name]'] = tags_dict['[Turb base name]']
            except KeyError:
                tags_dict['[turb_base_name]'] = None
                tags_dict['[Turb base name]'] = None
            tags_dict['[DLC]'] = tags_dict['[Case id.]'].split('_')[0][3:]
            tags_dict['[pbs_out_dir]'] = 'pbs_out/%s/' % dlc_case
            tags_dict['[pbs_in_dir]'] = 'pbs_in/%s/' % dlc_case
            tags_dict['[iter_dir]'] = 'iter/%s/' % dlc_case
            # the default spreadsheets do not define the tags related to the
            # eigen analsyis yet
            if '[eigen_analysis]' in tags_dict and tags_dict['[eigen_analysis]']:
                rpl = (dlc_case, tags_dict['[Case id.]'])
                if '[eigenfreq_dir]' in tags_dict:
                    tags_dict['[eigenfreq_dir]'] = 'res_eigen/%s/%s/' % rpl
            t_stop = float(tags_dict['[time_stop]'])
            t0 = float(tags_dict['[t0]'])
            tags_dict['[duration]'] = str(t_stop - t0)
            opt_tags.append(tags_dict.copy())

    return opt_tags


def read_tags_spreadsheet(fname):
    """Read a spreadsheet with HAWC2 tags, make sure no 0/1/nan ends up
    replacing the ";" or "" (empty). Do not add any other tags.

    Returns
    -------

    opt_tags : [{}, {}] list of dictionaries
    """

    df = pd.read_excel(fname)
    df2 = df.astype(str)
    opt_tags = []
    for count, row in df2.iterrows():
        tags_dict = {}
        # construct to dict, convert unicode keys/values to strings
        for key, value in row.items():
            if isinstance(value, str):
                tags_dict[str(key)] = str(value)
            else:
                tags_dict[str(key)] = value
            # convert ; and empty to False/True
            if tags_dict[str(key)] == ';':
                tags_dict[str(key)] = False
            elif tags_dict[str(key)] == '':
                tags_dict[str(key)] = True
            elif tags_dict[str(key)].lower() == 'nan':
                tags_dict[str(key)] = True
        opt_tags.append(tags_dict.copy())

    return opt_tags


class Tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fpath = os.path.join(os.path.dirname(__file__), 'data/DLCs')

    def test_read_tag_exchange_file(self):

        df_list = misc.read_excel_files(self.fpath, fext='xlsx', pignore=None,
                                        sheet=0, pinclude=None)

#        df = df_list[list(df_list.keys())[0]]
#        df.fillna('', inplace=True)
#        df.replace(';', False, inplace=True)

    def test_excel_stabcon(self):
        opt_tags = excel_stabcon(self.fpath)


if __name__ == '__main__':

    unittest.main()
