# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:00:25 2014

@author: dave
"""
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import matplotlib as mpl

from wetb.prepost import Simulations as sim
from wetb.prepost import (dlcdefs, dlcplots, windIO)
from wetb.prepost.simchunks import (create_chunks_htc_pbs, AppendDataFrames)
from wetb.prepost.GenerateDLCs import GenerateDLCCases

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
# plt.rc('text', usetex=True)

# import socket
# from sys import platform
# set runmethod based on the platform host
# if platform in ["linux", "linux2", "darwin"]:
#     RUNMETHOD = 'linux-script'
# elif platform == "win32":
#     RUNMETHOD = 'windows-script'
# else:
#     RUNMETHOD = 'none'
RUNMETHOD = 'none'
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
    master.tags['[seed]'] = None
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
    try:
        mt['[time_stop]'] = mt['[time stop]']
    except KeyError:
        mt['[time stop]'] = mt['[time_stop]']
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


def variable_tag_func_mod1(master, case_id_short=False):
    """
    CAUTION: this is version will add an additional layer in the folder
    structure in order to seperate input and output file types:
        input:
            htc
                DLCs
            data
            control
        output:
            res
            logfiles

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

    mt['[Case folder]'] = mt['[Case folder]'].lower()
    dlc_case = mt['[Case folder]']

    if '[Case id.]' in mt.keys():
        mt['[case_id]'] = mt['[Case id.]'].lower()
    if '[time stop]' in mt.keys():
        mt['[time_stop]'] = mt['[time stop]']
    else:
        mt['[time stop]'] = mt['[time_stop]']
    try:
        mt['[turb_base_name]'] = mt['[Turb base name]']
    except KeyError:
        mt['[turb_base_name]'] = None

    mt['[data_dir]'] = 'input/data/'
    mt['[res_dir]'] = 'output/res/%s/' % dlc_case
    mt['[log_dir]'] = 'output/logfiles/%s/' % dlc_case
    mt['[htc_dir]'] = 'input/htc/%s/' % dlc_case
    try:
        mt['[time_stop]'] = mt['[time stop]']
    except KeyError:
        mt['[time stop]'] = mt['[time_stop]']
    mt['[DLC]'] = mt['[Case id.]'].split('_')[0][3:]
    mt['[pbs_out_dir]'] = 'output/pbs_out/%s/' % dlc_case
    mt['[pbs_in_dir]'] = 'output/pbs_in/%s/' % dlc_case
    mt['[iter_dir]'] = 'output/iter/%s/' % dlc_case
    if '[eigen_analysis]' in mt and mt['[eigen_analysis]']:
        rpl = os.path.join(dlc_case, mt['[Case id.]'])
        mt['[eigenfreq_dir]'] = 'output/res_eigen/%s/' % rpl
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
                      runmethod=None, write_htc=True, zipchunks=False,
                      walltime='04:00:00', postpro_node=False, compress=False,
                      dlcs_dir='htc/DLCs', postpro_node_zipchunks=True,
                      wine_arch='win32', wine_prefix='.wine32', ppn=17,
                      m=[3,4,6,8,9,10,12], prelude='', linux=False,
                      update_model_data=False):
    """
    Launch load cases defined in Excel files
    """

    iter_dict = dict()
    iter_dict['[empty]'] = [False]

    if postpro_node or postpro_node_zipchunks:
        pyenv = 'py36-wetb'
    else:
        pyenv = None

    # FIXME: THIS IS VERY MESSY, we have wine_prefix/arch and exesingle/chunks
    if linux:
        wine_arch = None
        wine_prefix = None
        prelude = 'module load mpi/openmpi_1.6.5_intelv14.0.0\n'

    # if linux:
    #     pyenv = 'py36-wetb'
    #     pyenv_cmd = 'source /home/python/miniconda3/bin/activate'
    #     exesingle = "{hawc2_exe:} {fname_htc:}"
    #     exechunks = "({winenumactl:} {hawc2_exe:} {fname_htc:}) "
    #     exechunks += "2>&1 | tee {fname_pbs_out:}"
    # else:
    #     pyenv = ''
    #     pyenv_cmd = 'source /home/ozgo/bin/activate_hawc2cfd.sh'
    #     exesingle = "time {hawc2_exe:} {fname_htc:}"
    #     exechunks = "(time numactl --physcpubind=$CPU_NR {hawc2_exe:} {fname_htc:}) "
    #     exechunks += "2>&1 | tee {fname_pbs_out:}"

    # see if a htc/DLCs dir exists
    # Load all DLC definitions and make some assumptions on tags that are not
    # defined
    if os.path.exists(dlcs_dir):
        opt_tags = dlcdefs.excel_stabcon(dlcs_dir, silent=silent,
                                         p_source=P_SOURCE)
    else:
        opt_tags = dlcdefs.excel_stabcon(os.path.join(P_SOURCE, 'htc'),
                                         silent=silent, p_source=P_SOURCE)

    if len(opt_tags) < 1:
        raise ValueError('There are is not a single case defined. Make sure '
                         'the DLC spreadsheets are configured properly.')

    # add all the root files, except anything with *.zip
    f_ziproot = []
    for (dirpath, dirnames, fnames) in os.walk(P_SOURCE):
        # remove all zip files
        for i, fname in enumerate(fnames):
            if not fname.endswith('.zip'):
                f_ziproot.append(fname)
        break
    # and add those files
    for opt in opt_tags:
        opt['[zip_root_files]'] = f_ziproot

    if runmethod == None:
        runmethod = RUNMETHOD

    master = master_tags(sim_id, runmethod=runmethod, silent=silent,
                         verbose=verbose)
    master.tags['[sim_id]'] = sim_id
    master.tags['[walltime]'] = walltime
    master.output_dirs.append('[Case folder]')
    master.output_dirs.append('[Case id.]')

    # TODO: copy master and DLC exchange files to p_root too!!

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags vartag_func()
    # has precedense over iter_dict, which has precedence over opt_tags.
    # dlcdefs.vartag_excel_stabcon adds support for creating hydro files
    vartag_func = dlcdefs.vartag_excel_stabcon
    cases = sim.prepare_launch(iter_dict, opt_tags, master, vartag_func,
                               write_htc=write_htc, runmethod=runmethod,
                               copyback_turb=True, update_cases=False, msg='',
                               ignore_non_unique=False, run_only_new=False,
                               pbs_fname_appendix=False, short_job_names=False,
                               silent=silent, verbose=verbose, pyenv=pyenv,
                               m=[3,4,6,8,9,10,12], postpro_node=postpro_node,
                               exechunks=None, exesingle=None, prelude=prelude,
                               postpro_node_zipchunks=postpro_node_zipchunks,
                               wine_arch=wine_arch, wine_prefix=wine_prefix,
                               update_model_data=update_model_data)

    if pbs_turb:
        # to avoid confusing HAWC2 simulations and Mann64 generator PBS files,
        # MannTurb64 places PBS launch scripts in a "pbs_in_turb" folder
        mann64 = sim.MannTurb64(silent=silent)
        mann64.walltime = '00:59:59'
        mann64.queue = 'workq'
        mann64.gen_pbs(cases)

    if zipchunks:
        # create chunks
        # sort so we have minimal copying turb files from mimer to node/scratch
        # note that walltime here is for running all cases assigned to the
        # respective nodes. It is not walltime per case.
        sorts_on = ['[DLC]', '[Windspeed]']
        create_chunks_htc_pbs(cases, sort_by_values=sorts_on, queue='workq',
                              ppn=ppn, nr_procs_series=3, walltime='09:00:00',
                              chunks_dir='zip-chunks-jess', compress=compress,
                              wine_arch=wine_arch, wine_prefix=wine_prefix,
                              prelude=prelude, ppn_pbs=20)

    df = sim.Cases(cases).cases2df()
    df.to_excel(os.path.join(POST_DIR, sim_id + '.xlsx'))


def post_launch(sim_id, statistics=True, rem_failed=True, check_logs=True,
                force_dir=False, update=False, saveinterval=2000, csv=False,
                m=[3, 4, 6, 8, 9, 10, 12], neq=1e7, no_bins=46, int_env=False,
                years=20.0, fatigue=True, A=None, AEP=False, nx=300,
                save_new_sigs=False, envelopeturbine=False, envelopeblade=False,
                save_iter=False, pbs_failed_path=False):

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

    if force_dir:
        for case in cc.cases:
            cc.cases[case]['[post_dir]'] = POST_DIR
            cc.cases[case]['[run_dir]'] = force_dir

    if check_logs:
        cc.post_launch(save_iter=save_iter, pbs_failed_path=pbs_failed_path)
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

        # example for combination of signals
#        name = 'stress1'
#        expr = '[p1-p1-node-002-forcevec-z]*3 + [p1-p1-node-002-forcevec-y]'
#        add_sigs = {name:expr}

        # in addition, sim_id and case_id are always added by default
        # FIXME: HAS TO BE THE SAME AS required IN postpro_node_merge
        tags = ['[Case folder]', '[run_dir]', '[res_dir]', '[DLC]',
                '[wsp]', '[Windspeed]', '[wdir]']
        add = None
        # general statistics for all channels channel
        # set neq=None here to calculate 1Hz equivalent loads
        df_stats = cc.statistics(calc_mech_power=False, i0=i0, i1=i1,
                                 tags=tags, add_sensor=add, ch_fatigue=None,
                                 update=update, saveinterval=saveinterval,
                                 suffix=suffix, save_new_sigs=save_new_sigs,
                                 csv=csv, m=m, neq=None, no_bins=no_bins,
                                 chs_resultant=[], A=A, add_sigs={})
        # save channel list
        chans = df_stats['channel'].unique()
        chans.sort()
        fname = os.path.join(POST_DIR, '%s_unique-channel-names.csv' % sim_id)
        pd.DataFrame(chans).to_csv(fname)

    # annual energy production
    if AEP:
        # load the statistics in case they are missing
        if not statistics:
            df_stats, Leq_df, AEP_df = cc.load_stats()

        # CAUTION: depending on the type output, electrical power can be two
        # different things with the DTU Wind Energy Controller.
        # Either manually set ch_powe to the correct value or use simple
        # mechanism to figure out which one of two expected values it is.
        if 'DLL-2-inpvec-2' in df_stats['channel'].unique():
            ch_powe = 'DLL-2-inpvec-2'
        elif 'DLL-generator_servo-inpvec-2' in df_stats['channel'].unique():
            ch_powe = 'DLL-generator_servo-inpvec-2'

        df_AEP = cc.AEP(df_stats, csv=csv, update=update, save=True,
                        ch_powe=ch_powe)

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
        cc.envelopes(ch_list=ch_list, append='_blade', int_env=int_env, Nx=nx)

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
        cc.envelopes(ch_list=ch_list, append='_turbine', int_env=int_env, Nx=nx)

    if fatigue:
        # load the statistics in case they are missing
        if not statistics:
            df_stats, Leq_df, AEP_df = cc.load_stats()
        # life time equivalent load for all channels
        df_Leq = cc.fatigue_lifetime(df_stats, neq, csv=csv, update=update,
                                     years=years, save=True)

    return df_stats, df_AEP, df_Leq


def postpro_node_merge(tqdm=False, zipchunks=False):
    """With postpro_node each individual case has a .csv file for the log file
    analysis and a .csv file for the statistics tables. Merge all these single
    files into one table/DataFrame.

    When using the zipchunk approach, all log file analysis and statistics
    are grouped into tar archives in the prepost-data directory.

    Parameters
    ----------

    tqdm : boolean, default=False
        Set to True for displaying a progress bar (provided by the tqdm module)
        when merging all csv files into a single table/pd.DataFrame.

    zipchunks : boolean, default=False
        Set to True if merging post-processing files grouped into tar archives
        as generated by the zipchunks approach.

    """
    # -------------------------------------------------------------------------
    # MERGE POSTPRO ON NODE APPROACH INTO ONE DataFrame
    # -------------------------------------------------------------------------
    lf = windIO.LogFile()
    path_pattern = os.path.join(P_RUN, 'logfiles', '*', '*.csv')
    if zipchunks:
        path_pattern = os.path.join(POST_DIR, 'loganalysis_chnk*.tar.xz')
    csv_fname = '%s_ErrorLogs.csv' % sim_id
    fcsv = os.path.join(POST_DIR, csv_fname)
    mdf = AppendDataFrames(tqdm=tqdm)
    # individual log file analysis does not have header, make sure to include
    # a line for the header
    cols = mdf.txt2txt(fcsv, path_pattern, tarmode='r:xz', header=None,
                       header_fjoined=lf._header(), recursive=True)

    # FIXME: this is due to bug in log file analysis. What is going on here??
    # fix that some cases do not have enough columns
    with open(fcsv.replace('.csv', '2.csv'), 'w') as f1:
        with open(fcsv) as f2:
            for line in f2.readlines():
                # older versions had 95, columns, newer 103
                if len(line.split(';'))==96 or len(line.split(';'))==104:
                    line = line.replace(';0.00000000000;nan;-0.0000;',
                                        '0.00000000000;nan;-0.0000;')
                f1.write(line)

    # convert from CSV to DataFrame
    df = lf.csv2df(fcsv.replace('.csv', '2.csv'))
    df.to_hdf(fcsv.replace('.csv', '.h5'), 'table')
    # -------------------------------------------------------------------------
    path_pattern = os.path.join(P_RUN, 'res', '*', '*.csv')
    csv_fname = '%s_statistics.csv' % sim_id
    if zipchunks:
        path_pattern = os.path.join(POST_DIR, 'statsdel_chnk*.tar.xz')
    fcsv = os.path.join(POST_DIR, csv_fname)
    mdf = AppendDataFrames(tqdm=tqdm)
    # individual log file analysis does not have header, make sure to include
    # a line for the header
    cols = mdf.txt2txt(fcsv, path_pattern, tarmode='r:xz', header=0, sep=',',
                       header_fjoined=None, recursive=True, fname_col='[case_id]')
    # and convert to df: takes 2 minutes
    fdf = fcsv.replace('.csv', '.h5')
    store = pd.HDFStore(fdf, mode='w', format='table', complevel=9,
                        complib='zlib')
    colnames = cols.split(',')
    # the first column is the channel name but the header is emtpy
    assert colnames[0] == ''
    colnames[0] = 'channel'
    dtypes = {col:np.float64 for col in colnames}
    dtypes['channel'] = str
    dtypes['[case_id]'] = str
    # when using min_itemsize the column names should be valid variable names
    # mitemsize = {'channel':60, '[case_id]':60}
    mdf.csv2df_chunks(store, fcsv, chunksize=1000000, min_itemsize={}, sep=',',
                      colnames=colnames, dtypes=dtypes, header=0)
    store.close()
    # -------------------------------------------------------------------------
    # merge missing cols onto stats
    # FIXME: HAS TO BE THE SAME AS tags IN post_launch
    required = ['[DLC]', '[run_dir]', '[wdir]', '[Windspeed]', '[res_dir]',
                '[case_id]', '[Case folder]']
    df = pd.read_hdf(fdf, 'table')

    # FIXME: why do some cases have a leading ./ (but most do not)?
    sel = df['[case_id]'].str.startswith('./')
    df.loc[sel,'[case_id]'] = df.loc[sel,'[case_id]'].str.replace('./', '', 1)
    # df now has case_id as the path to the statistics file: res/dlc12_xxx/yyy
    # while df_tags will have just yyy as case_id
    tmp = df['[case_id]'].str.split('/', expand=True)
    df['[case_id]'] = tmp[tmp.columns[-1]]

    cc = sim.Cases(POST_DIR, sim_id)
    df_tags = cc.cases2df()
    df_stats = pd.merge(df, df_tags[required], on=['[case_id]'])
    # find out if we have some misalignment between generated cases and results
    # this could happen when we added new cases and removed others
    if len(df_stats) != len(df):
        print('failed to merge required tags, something is wrong!')
        # find out which cases we lost and why
        print('number of entries lost:', len(df)-len(df_stats))
        s_df = set(df['[case_id]'].unique())
        s_stats = set(df_stats['[case_id]'].unique())
        print('nr of channels:', len(df['channel'].unique()))
        msg = 'nr of case_ids lost:'
        print(msg, (len(df)-len(df_stats))/len(df['channel'].unique()))
        print('following case_ids have mysteriously disappeared:')
        missing = list(s_df-s_stats)
        print(missing)
        # save misalligned cases
        fname = os.path.join(POST_DIR, '%s_misallgined_cases.tsv' % sim_id)
        pd.DataFrame(missing).to_csv(fname, sep='\t')

    df_stats.to_hdf(fdf, 'table', mode='w')
    df_stats.to_csv(fdf.replace('.h5', '.csv'))

    # -------------------------------------------------------------------------
    # save channel list
    chans = df_stats['channel'].unique()
    chans.sort()
    fname = os.path.join(POST_DIR, '%s_unique-channel-names.csv' % sim_id)
    pd.DataFrame(chans).to_csv(fname)


def prepare_failed(compress=False, wine_arch='win32', wine_prefix='.wine32',
                   prelude='', zipchunks=False):

    cc = sim.Cases(POST_DIR, sim_id, rem_failed=False)
    df_tags = cc.cases2df()

    # -------------------------------------------------------------------------
    # find failed cases and create pbs_in_failed dir
    cc.find_failed(df_cases=df_tags)
    sim.copy_pbs_in_failedcases(cc.cases_fail, path=opt.pbs_failed_path)

    if zipchunks and len(cc.cases_fail) > 0:
        # and for chunks as well
        sorts_on = ['[DLC]', '[Windspeed]']
        create_chunks_htc_pbs(cc.cases_fail, sort_by_values=sorts_on,
                              ppn=17, nr_procs_series=3, walltime='09:00:00',
                              chunks_dir='zip-chunks-jess-fail', compress=compress,
                              wine_arch=wine_arch, wine_prefix=wine_prefix,
                              prelude=prelude, queue='windq', i0=1000)


if __name__ == '__main__':

    parser = ArgumentParser(description = "pre- or post-processes DLC's")
    parser.add_argument('--prep', action='store_true', default=False,
                        dest='prep', help='create htc, pbs, files')
    parser.add_argument('--check_logs', action='store_true', default=False,
                        dest='check_logs', help='check the log files')
    parser.add_argument('--failed', action='store_true', default=False,
                        dest='failed', help='Create new pbs_in files for all '
                        'failed cases. Combine with --zipchunks to also create '
                        'new zipchunks for the failed cases.')
    parser.add_argument('--pbs_failed_path', default='pbs_in_fail', type=str,
                        action='store', dest='pbs_failed_path',
                        help='Copy pbs launch files of the failed cases to a '
                        'new directory in order to prepare a re-run. Default '
                        'value: pbs_in_failed.')
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
    parser.add_argument('--neq', type=float, default=1e7, action='store',
                        dest='neq', help='Equivalent cycles Neq used for '
                                         'Leq fatigue lifetime calculations.')
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
    parser.add_argument('--int_env', default=False, action='store_true',
                        dest='int_env', help='Interpolate load envelopes to '
                        'a fixed number of nx angles.')
    parser.add_argument('--nx', type=int, default=300, action='store',
                        dest='nx', help='Number of angles to interpolate the '
                        'load envelopes to. Default=300.')
    parser.add_argument('--zipchunks', default=False, action='store_true',
                        dest='zipchunks', help='Create PBS launch files for'
                        'running in zip-chunk find+xargs mode.')
    parser.add_argument('--compress', default=False, action='store_true',
                        dest='compress', help='When running in zip-chunk mode,'
                        'compress log and results files into chunks.')
    parser.add_argument('--pbs_turb', default=False, action='store_true',
                        dest='pbs_turb', help='Create PBS launch files to '
                        'create the turbulence boxes in stand alone mode '
                        'using the 64-bit Mann turbulence box generator. '
                        'This can be usefull if your turbulence boxes are too '
                        'big for running in HAWC2 32-bit mode. Only works on '
                        'Jess.')
    parser.add_argument('--walltime', default='04:00:00', type=str,
                        action='store', dest='walltime', help='Queue walltime '
                        'for each case/pbs file, format: HH:MM:SS '
                        'Default: 04:00:00')
    parser.add_argument('--postpro_node', default=False, action='store_true',
                        dest='postpro_node', help='Perform the log analysis '
                        'and stats calculation on the node right after the '
                        'simulation has finished in single pbs mode.')
    parser.add_argument('--no_postpro_node_zipchunks', default=True,
                        action='store_false', dest='no_postpro_node_zipchunks',
                        help='Do NOT perform the log analysis '
                        'and stats calculation on the node right after the '
                        'simulation has finished in zipchunks mode. '
                        'The default is True, use this flag to deactivate.')
    parser.add_argument('--postpro_node_merge', default=False,
                        action='store_true', dest='postpro_node_merge',
                        help='Merge all individual statistics and log file '
                        'analysis .csv files into one table/pd.DataFrame. '
                        'Requires that htc files have been created with '
                        '--prep --postpro_node. Combine with --zipchunks when '
                        '--prep --zipchunks was used in for generating and '
                        'running all simulations.')
    parser.add_argument('--gendlcs', default=False, action='store_true',
                        help='Generate DLC exchange files based on master DLC '
                        'spreadsheet.')
    parser.add_argument('--dlcmaster', type=str, default='htc/DLCs.xlsx',
                        action='store', dest='dlcmaster',
                        help='Optionally define an other location of the '
                        'Master spreadsheet file location, default value is: '
                        'htc/DLCs.xlsx')
    parser.add_argument('--dlcfolder', type=str, default='htc/DLCs/',
                        action='store', dest='dlcfolder', help='Optionally '
                        'define an other destination folder location for the '
                        'generated DLC exchange files, default: htc/DLCs/')
    parser.add_argument('--wine_64bit', default=False, action='store_true',
                        dest='wine_64bit', help='Run wine in 64-bit mode. '
                        'Only works on Jess. Sets --wine_arch and '
                        '--wine_prefix to win64 and .wine respectively.')
    parser.add_argument('--wine_arch', action='store', default='win32', type=str,
                        dest='wine_arch', help='Set to win32 for 32-bit, and '
                        'win64 for 64-bit. 64-bit only works on Jess. '
                        'Defaults to win32.')
    parser.add_argument('--wine_prefix', action='store', default='.wine32',
                        type=str, dest='wine_prefix', help='WINEPREFIX: '
                        'Directory used by wineserver. Default .wine32')
    parser.add_argument('--linux', action='store_true', default=False,
                        dest='linux', help='Do not use wine. Implies that '
                        'wine_prefix and wine_arch is set to None.')
    opt = parser.parse_args()

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

    if opt.wine_64bit:
        opt.wine_arch, opt.wine_prefix = ('win64', '.wine')

    if opt.gendlcs:
        DLB = GenerateDLCCases()
        DLB.execute(filename=os.path.join(P_SOURCE, opt.dlcmaster),
                    folder=os.path.join(P_RUN, opt.dlcfolder))

    # create HTC files and PBS launch scripts (*.p)
    if opt.prep:
        print('Start creating all the htc files and pbs_in files...')
        launch_dlcs_excel(sim_id, silent=False, zipchunks=opt.zipchunks,
                          pbs_turb=opt.pbs_turb, walltime=opt.walltime,
                          postpro_node=opt.postpro_node, runmethod=RUNMETHOD,
                          dlcs_dir=os.path.join(P_SOURCE, opt.dlcfolder),
                          postpro_node_zipchunks=opt.no_postpro_node_zipchunks,
                          wine_arch=opt.wine_arch, wine_prefix=opt.wine_prefix,
                          compress=opt.compress, linux=opt.linux)
    # post processing: check log files, calculate statistics
    if opt.check_logs or opt.stats or opt.fatigue or opt.envelopeblade \
        or opt.envelopeturbine or opt.AEP:
        post_launch(sim_id, check_logs=opt.check_logs, update=False,
                    force_dir=P_RUN, saveinterval=2500, csv=opt.csv,
                    statistics=opt.stats, years=opt.years, neq=opt.neq,
                    fatigue=opt.fatigue, A=opt.rotarea, AEP=opt.AEP,
                    no_bins=opt.no_bins, pbs_failed_path=opt.pbs_failed_path,
                    save_new_sigs=opt.save_new_sigs, save_iter=False,
                    envelopeturbine=opt.envelopeturbine, int_env=opt.int_env,
                    envelopeblade=opt.envelopeblade, nx=opt.nx)
    if opt.postpro_node_merge:
        postpro_node_merge(zipchunks=opt.zipchunks)
    if opt.failed:
        prepare_failed(zipchunks=opt.zipchunks, compress=opt.compress,
                       wine_arch=opt.wine_arch, wine_prefix=opt.wine_prefix)
    if opt.dlcplot:

        # simply plot all channels
        # fname = os.path.join(POST_DIR, '%s_unique-channel-names.csv' % sim_id)
        # df = pd.read_csv(fname, header=0, names=['c1', 'c2'])
        # plot_chans = {}
        # for i, k in enumerate(df['c2'].values.tolist()):
        #     plot_chans['i=%i' % i] = [k]

        plot_chans = {}
        plot_chans['$B1_{flap}$'] = ['setbeta-bladenr-1-flapnr-1']
        plot_chans['$B2_{flap}$'] = ['setbeta-bladenr-2-flapnr-1']
        plot_chans['$B3_{flap}$'] = ['setbeta-bladenr-3-flapnr-1']

        for comp in list('xyz'):
            chans = []
            node_nr = 3
            node_lab = 'root'
            # combine blades 1,2,3 stats into a single plot
            for nr in [1, 2, 3]:
                rpl = (nr, nr, node_nr, comp)
                chans.append('blade%i-blade%i-node-%03i-momentvec-%s' % rpl)
            plot_chans['$M_%s B123_{%s}$' % (comp, node_lab)] = chans

        chans = []
        # combine blade 1,2,3 pitch angle stats into a single plot
        for nr in [1, 2, 3]:
            chans.append('bearing-pitch%i-angle-deg' % nr)
        plot_chans['$B123_{pitch}$'] = chans

        plot_chans['RPM'] = ['bearing-shaft_rot-angle_speed-rpm']
        plot_chans['$P_e$'] = ['DLL-generator_servo-inpvec-2']
        plot_chans['$P_{mech}$'] = ['stats-shaft-power']
        plot_chans['$B3 U_y$'] = ['global-blade3-elem-018-zrel-1.00-State pos-y']
        plot_chans['$M_x T_B$'] = ['tower-tower-node-001-momentvec-x']
        plot_chans['$M_y T_B$'] = ['tower-tower-node-001-momentvec-y']
        plot_chans['$M_z T_B$'] = ['tower-tower-node-001-momentvec-z']
        plot_chans['$M_z T_T$'] = ['tower-tower-node-008-momentvec-z']
        plot_chans['$M_y Shaft_{MB}$'] = ['shaft-shaft-node-004-momentvec-y']
        plot_chans['$M_x Shaft_{MB}$'] = ['shaft-shaft-node-004-momentvec-x']
        plot_chans['$M_z Shaft_{MB}$'] = ['shaft-shaft-node-004-momentvec-z']

        sim_ids = [sim_id]
        figdir = os.path.join(POST_DIR, 'dlcplots/')
        dlcplots.plot_stats2(sim_ids, [POST_DIR], plot_chans,
                             fig_dir_base=figdir)
