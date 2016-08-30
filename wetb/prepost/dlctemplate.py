# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:00:25 2014

@author: dave
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from builtins import str
from builtins import range
from future import standard_library
standard_library.install_aliases()

import os
import socket
from argparse import ArgumentParser

#import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
#import matplotlib as mpl

from wetb.prepost import Simulations as sim
from wetb.prepost import dlcdefs
from wetb.prepost import dlcplots
from wetb.prepost.simchunks import create_chunks_htc_pbs

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
# on Gorm tex printing doesn't work
if socket.gethostname()[:2] == 'g-':
    RUNMETHOD = 'gorm'
elif socket.gethostname()[:4] == 'jess':
    RUNMETHOD = 'jess'
else:
    plt.rc('text', usetex=True)
plt.rc('legend', fontsize=11)
plt.rc('legend', numpoints=1)
plt.rc('legend', borderaxespad=0)

# =============================================================================
### MODEL
# =============================================================================

def master_tags(sim_id, runmethod='local', silent=False, verbose=False):
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

    # TODO: write a lot of logical tests for the tags!!
    # TODO: tests to check if the dirs are setup properly (ending slahses ...)
    # FIXME: some tags are still variable! Only static tags here that do
    # not depent on any other variable that can change

    master = sim.HtcMaster(verbose=verbose, silent=silent)
    # set the default tags
    master = dlcdefs.tags_defaults(master)

    # =========================================================================
    # SOURCE FILES
    # =========================================================================
#    # TODO: move to variable_tag
#    rpl = (p_root, project, sim_id)
#    if runmethod in ['local', 'local-script', 'none', 'local-ram']:
#        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
#    elif runmethod == 'windows-script':
#        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
#    elif runmethod == 'gorm':
#        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
#    elif runmethod == 'jess':
#        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
#    else:
#        msg='unsupported runmethod, options: none, local, gorm or opt'
#        raise ValueError, msg

    master.tags['[master_htc_file]'] = MASTERFILE
    master.tags['[master_htc_dir]'] = P_MASTERFILE
    # directory to data, htc, SOURCE DIR
    if P_SOURCE[-1] == os.sep:
        master.tags['[model_dir_local]']  = P_SOURCE
    else:
        master.tags['[model_dir_local]']  = P_SOURCE + os.sep
    if P_RUN[-1] == os.sep:
        master.tags['[run_dir]'] = P_RUN
    else:
        master.tags['[run_dir]'] = P_RUN + os.sep

    master.tags['[post_dir]'] = POST_DIR
    master.tags['[sim_id]'] = sim_id
    # set the model_zip tag to include the sim_id
    master.tags['[model_zip]'] = PROJECT
    master.tags['[model_zip]'] += '_' + master.tags['[sim_id]'] + '.zip'
    # -------------------------------------------------------------------------
    # FIXME: this is very ugly. We should read default values set in the htc
    # master file with the HAWC2Wrapper !!
    # default tags turbulence generator (required for 64-bit Mann generator)
    # alfaeps, L, gamma, seed, nr_u, nr_v, nr_w, du, dv, dw high_freq_comp
    master.tags['[MannAlfaEpsilon]'] = 1.0
    master.tags['[MannL]'] = 29.4
    master.tags['[MannGamma]'] = 3.0
    master.tags['[tu_seed]'] = 0
    master.tags['[turb_nr_u]'] = 8192
    master.tags['[turb_nr_v]'] = 32
    master.tags['[turb_nr_w]'] = 32
    master.tags['[turb_dx]'] = 1
    master.tags['[turb_dy]'] = 6.5
    master.tags['[turb_dz]'] = 6.5
    master.tags['[high_freq_comp]'] = 1
    # -------------------------------------------------------------------------

    return master


def variable_tag_func(master, case_id_short=False):
    """
    Function which updates HtcMaster.tags and returns an HtcMaster object

    Only use lower case characters for case_id since a hawc2 result and
    logfile are always in lower case characters.

    BE CAREFULL: if you change a master tag that is used to dynamically
    calculate an other tag, that change will be propageted over all cases,
    for example:
    master.tags['tag1'] *= master.tags[tag2]*master.tags[tag3']
    it will accumlate over each new case. After 20 cases
    master.tags['tag1'] = (master.tags[tag2]*master.tags[tag3'])^20
    which is not wanted, you should do
    master.tags['tag1'] = tag1_base*master.tags[tag2]*master.tags[tag3']

    This example is based on reading the default DLC spreadsheets, and is
    already included in the dlcdefs.excel_stabcon
    """

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
        rpl = os.path.join(dlc_case, mt['[Case id.]'])
        mt['[eigenfreq_dir]'] = 'res_eigen/%s/' % rpl
    mt['[duration]'] = str(float(mt['[time_stop]']) - float(mt['[t0]']))
    # replace nan with empty
    for ii, jj in mt.items():
        if jj == 'nan':
            mt[ii] = ''

    return master

# =============================================================================
### PRE- POST
# =============================================================================

def launch_dlcs_excel(sim_id, silent=False, verbose=False, pbs_turb=False,
                      runmethod=None, write_htc=True, zipchunks=False):
    """
    Launch load cases defined in Excel files
    """

    iter_dict = dict()
    iter_dict['[empty]'] = [False]

    # see if a htc/DLCs dir exists
    dlcs_dir = os.path.join(P_SOURCE, 'htc', 'DLCs')
    if os.path.exists(dlcs_dir):
        opt_tags = dlcdefs.excel_stabcon(dlcs_dir, silent=silent)
    else:
        opt_tags = dlcdefs.excel_stabcon(os.path.join(P_SOURCE, 'htc'),
                                         silent=silent)

    if len(opt_tags) < 1:
        raise ValueError('There are is not a single case defined. Make sure '
                         'the DLC spreadsheets are configured properly.')

    # add all the root files, except anything with *.zip
    f_ziproot = []
    for (dirpath, dirnames, fnames) in os.walk(P_SOURCE):
        # remove all zip files
        for i, fname in enumerate(fnames):
            if fname.endswith('.zip'):
                fnames.pop(i)
        f_ziproot.extend(fnames)
        break
    # and add those files
    for opt in opt_tags:
        opt['[zip_root_files]'] = f_ziproot

    if runmethod == None:
        runmethod = RUNMETHOD

    master = master_tags(sim_id, runmethod=runmethod, silent=silent,
                         verbose=verbose)
    master.tags['[sim_id]'] = sim_id
    master.output_dirs.append('[Case folder]')
    master.output_dirs.append('[Case id.]')

    # TODO: copy master and DLC exchange files to p_root too!!

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    # variable_tag_func() has precedense over iter_dict, which has precedence
    # over opt_tags. So opt_tags comes last
    # variable_tag func is not required because everything is already done
    # in dlcdefs.excel_stabcon
    no_variable_tag_func = None
    cases = sim.prepare_launch(iter_dict, opt_tags, master, no_variable_tag_func,
                               write_htc=write_htc, runmethod=runmethod,
                               copyback_turb=True, update_cases=False, msg='',
                               ignore_non_unique=False, run_only_new=False,
                               pbs_fname_appendix=False, short_job_names=False,
                               silent=silent, verbose=verbose, pyenv=None)

    if pbs_turb:
        # to avoid confusing HAWC2 simulations and Mann64 generator PBS files,
        # MannTurb64 places PBS launch scripts in a "pbs_in_turb" folder
        mann64 = sim.MannTurb64(silent=silent)
        mann64.gen_pbs(cases)

    if zipchunks:
        # create chunks
        # sort so we have minimal copying turb files from mimer to node/scratch
        sorts_on = ['[DLC]', '[Windspeed]']
        create_chunks_htc_pbs(cases, sort_by_values=sorts_on, ppn=20,
                              nr_procs_series=9, processes=1,
                              walltime='20:00:00', chunks_dir='zip-chunks-jess')
        create_chunks_htc_pbs(cases, sort_by_values=sorts_on, ppn=12,
                              nr_procs_series=15, processes=1,
                              walltime='20:00:00', chunks_dir='zip-chunks-gorm')


def post_launch(sim_id, statistics=True, rem_failed=True, check_logs=True,
                force_dir=False, update=False, saveinterval=2000, csv=False,
                m=[1, 3, 4, 5, 6, 8, 10, 12, 14], neq=None, no_bins=46,
                years=20.0, fatigue=True, A=None, AEP=False,
                save_new_sigs=False, envelopeturbine=False, envelopeblade=False,
                save_iter=False):

    # =========================================================================
    # check logfiles, results files, pbs output files
    # logfile analysis is written to a csv file in logfiles directory
    # =========================================================================
    # load the file saved in post_dir
    config = {}
    config['Weibull'] = {}
    config['Weibull']['Vr'] = 11.
    config['Weibull']['Vref'] = 50
    config['nn_shaft'] = 4
    cc = sim.Cases(POST_DIR, sim_id, rem_failed=rem_failed, config=config)
    cc.force_lower_case_id()

    if force_dir:
        for case in cc.cases:
            cc.cases[case]['[post_dir]'] = POST_DIR
            cc.cases[case]['[run_dir]'] = force_dir

    if check_logs:
        cc.post_launch(save_iter=save_iter)
    elif rem_failed:
        cc.remove_failed()

    # using suffix is only relevant if we have more cases then the save interval
    if len(cc.cases) > saveinterval:
        suffix = True
    else:
        suffix = False

    df_stats, df_AEP, df_Leq = None, None, None

    if statistics:
        i0, i1 = 0, -1

        # in addition, sim_id and case_id are always added by default
        tags = ['[Case folder]']
        add = None
        # general statistics for all channels channel
        df_stats = cc.statistics(calc_mech_power=True, i0=i0, i1=i1,
                                 tags=tags, add_sensor=add, ch_fatigue=None,
                                 update=update, saveinterval=saveinterval,
                                 suffix=suffix, save_new_sigs=save_new_sigs,
                                 csv=csv, m=m, neq=neq, no_bins=no_bins,
                                 chs_resultant=[], A=A)
        # annual energy production
        if AEP:
            df_AEP = cc.AEP(df_stats, csv=csv, update=update, save=True)

    if envelopeblade:
        ch_list = []
        for iblade in range(1, 4):
            for i in range(1, 18):
                rpl = (iblade, iblade, i)
                ch_list.append(['blade%i-blade%i-node-%3.3i-momentvec-x' % rpl,
                                'blade%i-blade%i-node-%3.3i-momentvec-y' % rpl,
                                'blade%i-blade%i-node-%3.3i-momentvec-z' % rpl,
                                'blade%i-blade%i-node-%3.3i-forcevec-x' % rpl,
                                'blade%i-blade%i-node-%3.3i-forcevec-y' % rpl,
                                'blade%i-blade%i-node-%3.3i-forcevec-z' % rpl])
        cc.envelope(ch_list=ch_list, append='_blade')

    if envelopeturbine:
        ch_list = [['tower-tower-node-001-momentvec-x',
                    'tower-tower-node-001-momentvec-y',
                    'tower-tower-node-001-momentvec-z'],
                   ['tower-tower-node-022-momentvec-x',
                   'tower-tower-node-022-momentvec-y',
                   'tower-tower-node-022-momentvec-z',
                   'tower-tower-node-022-forcevec-x',
                   'tower-tower-node-022-forcevec-y',
                   'tower-tower-node-022-forcevec-z'],
                   ['hub1-hub1-node-001-momentvec-x',
                   'hub1-hub1-node-001-momentvec-y',
                   'hub1-hub1-node-001-momentvec-z']]
        cc.envelope(ch_list=ch_list, append='_turbine')
    if fatigue:
        # load the statistics in case they are missing
        if not statistics:
            df_stats, Leq_df, AEP_df = cc.load_stats()
        # life time equivalent load for all channels
        df_Leq = cc.fatigue_lifetime(df_stats, neq, csv=csv, update=update,
                                     years=years, save=True)

    return df_stats, df_AEP, df_Leq


if __name__ == '__main__':

    parser = ArgumentParser(description = "pre- or post-processes DLC's")
    parser.add_argument('--prep', action='store_true', default=False,
                        dest='prep', help='create htc, pbs, files')
    parser.add_argument('--check_logs', action='store_true', default=False,
                        dest='check_logs', help='check the log files')
    parser.add_argument('--stats', action='store_true', default=False,
                        dest='stats', help='calculate statistics and 1Hz '
                                           'equivalent loads')
    parser.add_argument('--fatigue', action='store_true', default=False,
                        dest='fatigue', help='calculate Leq for a full DLC')
    parser.add_argument('--AEP', action='store_true', default=False,
                        dest='AEP', help='calculate AEP, requires '
                        'htc/DLCs/dlc_config.xlsx')
    parser.add_argument('--csv', action='store_true', default=False,
                        dest='csv', help='Save data also as csv file')
    parser.add_argument('--years', type=float, default=20.0, action='store',
                        dest='years', help='Total life time in years')
    parser.add_argument('--no_bins', type=float, default=46.0, action='store',
                        dest='no_bins', help='Number of bins for fatigue loads')
    parser.add_argument('--neq', type=float, default=None, action='store',
                        dest='neq', help='Equivalent cycles neq, default 1 Hz '
                                         'equivalent load (neq = simulation '
                                         'duration in seconds)')
    parser.add_argument('--rotarea', type=float, default=None, action='store',
                        dest='rotarea', help='Rotor area for C_T, C_P')
    parser.add_argument('--save_new_sigs', default=False, action='store_true',
                        dest='save_new_sigs', help='Save post-processed sigs')
    parser.add_argument('--dlcplot', default=False, action='store_true',
                        dest='dlcplot', help='Plot DLC load basis results')
    parser.add_argument('--envelopeblade', default=False, action='store_true',
                        dest='envelopeblade', help='Compute envelopeblade')
    parser.add_argument('--envelopeturbine', default=False, action='store_true',
                        dest='envelopeturbine', help='Compute envelopeturbine')
    parser.add_argument('--zipchunks', default=False, action='store_true',
                        dest='zipchunks', help='Create PBS launch files for'
                        'running in zip-chunk find+xargs mode.')
    parser.add_argument('--pbs_turb', default=False, action='store_true',
                        dest='pbs_turb', help='Create PBS launch files to '
                        'create the turbulence boxes in stand alone mode '
                        'using the 64-bit Mann turbulence box generator. '
                        'This can be usefull if your turbulence boxes are too '
                        'big for running in HAWC2 32-bit mode. Only works on '
                        'Jess. ')
    opt = parser.parse_args()

    # TODO: use arguments to determine the scenario:
    # --plots, --report, --...

    # -------------------------------------------------------------------------
#    # manually configure paths, HAWC2 model root path is then constructed as
#    # p_root_remote/PROJECT/sim_id, and p_root_local/PROJECT/sim_id
#    # adopt accordingly when you have configured your directories differently
#    p_root_remote = '/mnt/hawc2sim/'
#    p_root_local = '/mnt/hawc2sim/'
#    # project name, sim_id, master file name
#    PROJECT = 'demo'
#    sim_id = 'A0001'
#    MASTERFILE = 'dtu10mw_avatar_master_A0001.htc'
#    # MODEL SOURCES, exchanche file sources
#    P_RUN = os.path.join(p_root_remote, PROJECT, sim_id+'/')
#    P_SOURCE = os.path.join(p_root_local, PROJECT, sim_id)
#    # location of the master file
#    P_MASTERFILE = os.path.join(p_root_local, PROJECT, sim_id, 'htc', '_master/')
#    # location of the pre and post processing data
#    POST_DIR = os.path.join(p_root_remote, PROJECT, sim_id, 'prepost-data/')
#    force_dir = P_RUN
#    launch_dlcs_excel(sim_id)
#    post_launch(sim_id, check_logs=True, update=False, force_dir=force_dir,
#                saveinterval=2000, csv=True, fatigue_cycles=True, fatigue=False)
    # -------------------------------------------------------------------------

    # auto configure directories: assume you are running in the root of the
    # relevant HAWC2 model
    # and assume we are in a simulation case of a certain turbine/project
    P_RUN, P_SOURCE, PROJECT, sim_id, P_MASTERFILE, MASTERFILE, POST_DIR \
        = dlcdefs.configure_dirs(verbose=True)

    # create HTC files and PBS launch scripts (*.p)
    if opt.prep:
        print('Start creating all the htc files and pbs_in files...')
        launch_dlcs_excel(sim_id, silent=False, zipchunks=opt.zipchunks,
                          pbs_turb=opt.pbs_turb)
    # post processing: check log files, calculate statistics
    if opt.check_logs or opt.stats or opt.fatigue or opt.envelopeblade or opt.envelopeturbine:
        post_launch(sim_id, check_logs=opt.check_logs, update=False,
                    force_dir=P_RUN, saveinterval=2000, csv=opt.csv,
                    statistics=opt.stats, years=opt.years, neq=opt.neq,
                    fatigue=opt.fatigue, A=opt.rotarea, AEP=opt.AEP,
                    no_bins=opt.no_bins, nn_blr=opt.nn_blr, nn_twt=opt.nn_twt,
                    save_new_sigs=opt.save_new_sigs, save_iter=False,
                    envelopeturbine=opt.envelopeturbine,
                    envelopeblade=opt.envelopeblade)
    if opt.dlcplot:
        sim_ids = [sim_id]
        figdir = os.path.join(POST_DIR, 'figures/%s' % '-'.join(sim_ids))
        dlcplots.plot_stats2(sim_ids, [POST_DIR], fig_dir_base=figdir)
