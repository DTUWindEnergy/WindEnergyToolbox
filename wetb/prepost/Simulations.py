# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:16:34 2011

@author: dave
__author__ = "David Verelst <dave@dtu.dk>"
__license__ = "GPL-2+"
"""
# standard python library
import os
import subprocess as sproc
import copy
import zipfile
import shutil
import datetime
import math
import pickle
import re
# what is actually the difference between warnings and logging.warn?
# for which context is which better?
import warnings
import logging
from operator import itemgetter
from time import time
#import Queue
#import threading
#from multiprocessing import Pool

import numpy as np
import scipy
#import matplotlib.pyplot as plt
import pandas as pd
import tables as tbl

# custom libraries
from wetb.fatigue_tools.bearing_damage import bearing_damage
from wetb.prepost import misc
from wetb.prepost import windIO
from wetb.prepost import prepost
from wetb.dlc import high_level as dlc
#from wetb.prepost.GenerateHydro import hydro_input
from wetb.utils.envelope import compute_envelope

#def join_path(*args):
#    return os_path_join(*args).replace("\\","/")
#os.path.join = join_path



def load_pickled_file(source):
    FILE = open(source, 'rb')
    result = pickle.load(FILE)
    FILE.close()
    return result

def save_pickle(source, variable):
    FILE = open(source, 'wb')
    pickle.dump(variable, FILE, protocol=2)
    FILE.close()

def write_file(file_path, file_contents, mode):
    """
    INPUT:
        file_path: path/to/file/name.csv
        string   : file contents is a string
        mode     : reading (r), writing (w), append (a),...
    """

    FILE = open(file_path, mode)
    FILE.write(file_contents)
    FILE.close()

def create_multiloop_list(iter_dict, debug=False):
    """
    Create a list based on multiple nested loops
    ============================================

    Considerd the following example

    >>> for v in range(V_start, V_end, V_delta):
    ...     for y in range(y_start, y_end, y_delta):
    ...         for c in range(c_start, c_end, c_delta):
    ...             print v, y, c

    Could be replaced by a list with all these combinations. In order to
    replicate this with create_multiloop_list, iter_dict should have
    the following structure

    >>> iter_dict = dict()
    >>> iter_dict['v'] = range(V_start, V_end, V_delta)
    >>> iter_dict['y'] = range(y_start, y_end, y_delta)
    >>> iter_dict['c'] = range(c_start, c_end, c_delta)
    >>> iter_list = create_multiloop_list(iter_dict)
    >>> for case in iter_list:
    ...     print case['v'], case['y'], case['c']

    Parameters
    ----------

    iter_dict : dictionary
        Key holds a valid tag as used in HtcMaster.tags. The corresponding
        value shouuld be a list of values to be considered.

    Output
    ------

    iter_list : list
        List containing dictionaries. Each entry is a combination of the
        given iter_dict keys.

    Example
    -------

    >>> iter_dict={'[wind]':[5,6,7],'[coning]':[0,-5,-10]}
    >>> create_multiloop_list(iter_dict)
    [{'[wind]': 5, '[coning]': 0},
     {'[wind]': 5, '[coning]': -5},
     {'[wind]': 5, '[coning]': -10},
     {'[wind]': 6, '[coning]': 0},
     {'[wind]': 6, '[coning]': -5},
     {'[wind]': 6, '[coning]': -10},
     {'[wind]': 7, '[coning]': 0},
     {'[wind]': 7, '[coning]': -5},
     {'[wind]': 7, '[coning]': -10}]
    """

    iter_list = []

    # fix the order of the keys
    key_order = list(iter_dict.keys())
    nr_keys = len(key_order)
    nr_values,indices = [],[]
    # determine how many items on each key
    for key in key_order:
        # each value needs to be an iterable! len() will fail if it isn't
        # count how many values there are for each key
        if type(iter_dict[key]).__name__ != 'list':
            print('%s does not hold a list' % key)
            raise ValueError('Each value in iter_dict has to be a list!')
        nr_values.append(len(iter_dict[key]))
        # create an initial indices list
        indices.append(0)

    if debug: print(nr_values, indices)

    go_on = True
    # keep track on which index you are counting, start at the back
    loopkey = nr_keys -1
    cc = 0
    while go_on:
        if debug: print(indices)

        # Each entry on the list is a dictionary with the parameter combination
        iter_list.append(dict())

        # save all the different combination into one list
        for keyi in range(len(key_order)):
            key = key_order[keyi]
            # add the current combination of values as one dictionary
            iter_list[cc][key] = iter_dict[key][indices[keyi]]

        # +1 on the indices of the last entry, the overflow principle
        indices[loopkey] += 1

        # cycle backwards thourgh all dimensions and propagate the +1 if the
        # current dimension is full. Hence overflow.
        for k in range(loopkey,-1,-1):
            # if the current dimension is over its max, set to zero and change
            # the dimension of the next. Remember we are going backwards
            if not indices[k] < nr_values[k] and k > 0:
                # +1 on the index of the previous dimension
                indices[k-1] += 1
                # set current loopkey index back to zero
                indices[k] = 0
                # if the previous dimension is not on max, break out
                if indices[k-1] < nr_values[k-1]:
                    break
            # if we are on the last dimension, break out if that is also on max
            elif k == 0 and not indices[k] < nr_values[k]:
                if debug: print(cc)
                go_on = False

        # fail safe exit mechanism...
        if cc > 20000:
            raise UserWarning('multiloop_list has already '+str(cc)+' items..')
            go_on = False

        cc += 1

    return iter_list

def local_shell_script(htc_dict, sim_id):
    """
    """
    shellscript = ''
    breakline = '"' + '*'*80 + '"'
    nr_cases = len(htc_dict)
    nr = 1
    for case in htc_dict:
        shellscript += 'echo ""' + '\n'
        shellscript += 'echo ' + breakline + '\n' + 'echo '
        shellscript += '" ===> Progress:'+str(nr)+'/'+str(nr_cases)+'"\n'
        # get a shorter version for the current cases tag_dict:
        scriptpath = os.path.join(htc_dict[case]['[run_dir]'], 'runall.sh')
        try:
            hawc2_exe = htc_dict[case]['[hawc2_exe]']
        except KeyError:
            hawc2_exe = 'hawc2mb.exe'
        htc_dir = htc_dict[case]['[htc_dir]']
        # log all warning messages: WINEDEBUG=-all!
        wine = 'WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        htc_target = os.path.join(htc_dir, case)
        shellscript += '%s %s %s \n' % (wine, hawc2_exe, htc_target)
        shellscript += 'echo ' + breakline + '\n'
        nr+=1

    write_file(scriptpath, shellscript, 'w')
    print('\nrun local shell script written to:')
    print(scriptpath)

def local_windows_script(cases, sim_id, nr_cpus=2):
    """
    """

    tot_cases = len(cases)
    i_script = 1
    i_case_script = 1
    cases_per_script = int(math.ceil(float(tot_cases)/float(nr_cpus)))
    # header of the new script, each process has its own copy
    header = ''
    header += 'rem\nrem\n'
    header += 'mkdir _%i_\n'
    # copy the data folder in case it holds a lot of .dat files
    header += 'robocopy .\data .\_%i_\data /e \n'
    # do not copy the following stuff
    exc_file_pat = ['*.log', '*.dat', '*.sel', '*.xls*', '*.bat']
    exc_dir_pat = ['_*_', 'data']
    header += 'robocopy .\ .\_%i_ /e '
    header += (' /xf ' + ' /xf '.join(exc_file_pat))
    header += (' /xd ' + ' /xd '.join(exc_dir_pat))
    header += '\n'
    header += 'cd _%i_\n'
    header += 'rem\nrem\n'
    footer = ''
    footer += 'rem\nrem\n'
    footer += 'cd ..\n'
    footer += 'robocopy .\_%i_\ /e .\ /move\n'
    footer += 'rem\nrem\n'
    shellscript = header % (i_script, i_script, i_script, i_script)

    stop = False

    for i_case, (cname, case) in enumerate(cases.items()):
#    for i_case, case in enumerate(sorted(cases.keys())):

        shellscript += 'rem\nrem\n'
        shellscript += 'rem ===> Progress: %3i / %3i\n' % (i_case+1, tot_cases)
        # copy turbulence from data base, if applicable
        if case['[turb_db_dir]'] is not None:
            # we are one dir up in cpu exe dir
            turb = case['[turb_base_name]'] + '*.bin'
            dbdir = os.path.join('./../', case['[turb_db_dir]'], turb)
            dbdir = dbdir.replace('/', '\\')
            rpl = (dbdir, case['[turb_dir]'].replace('/', '\\'))
            shellscript += 'copy %s %s\n' % rpl

        # get a shorter version for the current cases tag_dict:
        scriptpath = '%srunall-%i.bat' % (case['[run_dir]'], i_script)
        htcpath = case['[htc_dir]'][:-1].replace('/', '\\') # ditch the /
        try:
            hawc2_exe = case['[hawc2_exe]']
        except KeyError:
            hawc2_exe = 'hawc2mb.exe'
        rpl = (hawc2_exe.replace('/', '\\'), htcpath, cname.replace('/', '\\'))
        shellscript += "%s .\\%s\\%s\n" % rpl
        # copy back to data base directory if they do not exists there
        # remove turbulence file again, if copied from data base
        if case['[turb_db_dir]'] is not None:
            # copy back if it does not exist in the data base
            # IF EXIST "c:\test\file.ext"  (move /y "C:\test\file.ext" "C:\quality\" )
            turbu = case['[turb_base_name]'] + 'u.bin'
            turbv = case['[turb_base_name]'] + 'v.bin'
            turbw = case['[turb_base_name]'] + 'w.bin'
            dbdir = os.path.join('./../', case['[turb_db_dir]'])
            for tu in (turbu, turbv, turbw):
                tu_db = os.path.join(dbdir, tu).replace('/', '\\')
                tu_run = os.path.join(case['[turb_dir]'], tu).replace('/', '\\')
                rpl = (tu_db, tu_run, dbdir.replace('/', '\\'))
                shellscript += 'IF NOT EXIST "%s" move /y "%s" "%s"\n' % rpl
            # remove turbulence from run dir
            allturb = os.path.join(case['[turb_dir]'], '*.*')
            allturb = allturb.replace('/', '\\')
            # do not prompt for delete confirmation: /Q
            shellscript += 'del /Q "%s"\n' % allturb

        if i_case_script >= cases_per_script:
            # footer: copy all files back
            shellscript += footer % i_script
            stop = True
            write_file(scriptpath, shellscript, 'w')
            print('\nrun local shell script written to:')
            print(scriptpath)

            # header of the new script, each process has its own copy
            # but only if there are actually jobs left
            if i_case+1 < tot_cases:
                i_script += 1
                i_case_script = 1
                shellscript = header % (i_script, i_script, i_script, i_script)
                stop = False
        else:
            i_case_script += 1

    # we might have missed the footer of a partial script
    if not stop:
        shellscript += footer % i_script
        write_file(scriptpath, shellscript, 'w')
        print('\nrun local shell script written to:')
        print(scriptpath)

def run_local_ram(cases, check_log=True):

    ram_root = '/tmp/HAWC2/'

    if not os.path.exists(ram_root):
        os.makedirs(ram_root)

    print('copying data from run_dir to RAM...', end='')

    # first copy everything to RAM
    for ii, case in enumerate(cases):
        # all tags for the current case
        tags = cases[case]
        run_dir = copy.copy(tags['[run_dir]'])
        run_dir_ram = ram_root + tags['[sim_id]']
        if not os.path.exists(run_dir_ram):
            os.makedirs(run_dir_ram)
        # and also change the run dir so we can launch it easily
        tags['[run_dir]'] = run_dir_ram + '/'
        for root, dirs, files in os.walk(run_dir):
            run_dir_base = os.path.commonprefix([root, run_dir])
            cdir = root.replace(run_dir_base, '')
            dstbase = os.path.join(run_dir_ram, cdir)
            if not os.path.exists(dstbase):
                os.makedirs(dstbase)
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(dstbase, fname)
                shutil.copy2(src, dst)

    print('done')

    # launch from RAM
    run_local(cases, check_log=check_log)
    # change run_dir back to original
    for ii, case in enumerate(cases):
        tags = cases[case]
        tags['[run_dir]'] = run_dir

    print('copying data from RAM back to run_dir')
    print('run_dir: %s' % run_dir)

    # and copy everything back
    for root, dirs, files in os.walk(run_dir_ram):
        run_dir_base = os.path.commonprefix([root, run_dir_ram])
        cdir = root.replace(run_dir_base, '')
        # in case it is the same
        if len(cdir) == 0:
            pass
        # join doesn't work if cdir has a leading / ?? so drop it
        elif cdir[0] == '/':
            dstbase = os.path.join(run_dir, cdir[1:])
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(dstbase, fname)
            if not os.path.exists(dstbase):
                os.makedirs(dstbase)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print('src:', src)
                print('dst:', dst)
                print(e)
                print()
                pass

    print('...done')

    return cases

def run_local(cases, silent=False, check_log=True):
    """
    Run all HAWC2 simulations locally from cases
    ===============================================

    Run all case present in a cases dict locally and wait until HAWC2 is ready.

    In verbose mode, each HAWC2 simulation is also timed

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    check_log : boolean, default=False
        Check the log file emmidiately after execution of the HAWC2 case

    silent : boolean, default=False
        When False, usefull information will be printed and the HAWC2
        simulation time will be calculated from the Python perspective. The
        silent variable is also passed on to logcheck_case

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Update cases with the STDOUT of the respective HAWC2 simulation

    """

    # remember the current working directory
    cwd = str(os.getcwd())
    nr = len(cases)
    if not silent:
        print('')
        print('='*79)
        print('Be advised, launching %i HAWC2 simulation(s) sequentially' % nr)
        print('run dir: %s' % cases[list(cases.keys())[0]]['[run_dir]'])
        print('')

    if check_log:
        errorlogs = ErrorLogs(silent=silent)

    for ii, case in enumerate(cases):
        # all tags for the current case
        tags = cases[case]
        # for backward compatibility assume default HAWC2 executable
        try:
            hawc2_exe = tags['[hawc2_exe]']
        except KeyError:
            hawc2_exe = 'hawc2-latest'
        # TODO: if a turbulence data base is set, copy the files from there

        # the launch command
        cmd  = 'WINEDEBUG=-all WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        cmd += " %s %s%s" % (hawc2_exe, tags['[htc_dir]'], case)
        # remove any escaping in tags and case for security reasons
        cmd = cmd.replace('\\','')
        # browse to the correct launch path for the HAWC2 simulation
        os.chdir(tags['[run_dir]'])
        # create the required directories
        dirkeys = ['[data_dir]', '[htc_dir]', '[res_dir]', '[log_dir]',
                   '[eigenfreq_dir]', '[animation_dir]', '[turb_dir]',
                   '[micro_dir]', '[meander_dir]', '[opt_dir]', '[control_dir]',
                   '[mooring_dir]', '[hydro_dir]', '[externalforce]']
        for dirkey in dirkeys:
            if tags[dirkey]:
                if not os.path.exists(tags[dirkey]):
                    os.makedirs(tags[dirkey])

        if not silent:
            start = time()
            progress = '%4i/%i  : %s%s' % (ii+1, nr, tags['[htc_dir]'], case)
            print('*'*75)
            print(progress)

        # and launch the HAWC2 simulation
        p = sproc.Popen(cmd,stdout=sproc.PIPE,stderr=sproc.STDOUT,shell=True)

        # p.wait() will lock the current shell until p is done
        # p.stdout.readlines() checks if there is any output, but also locks
        # the thread if nothing comes back
        # save the output that HAWC2 sends to the shell to the cases
        # note that this is a list, each item holding a line
        cases[case]['sim_STDOUT'] = p.stdout.readlines()
        # wait until HAWC2 finished doing its magic
        p.wait()

        if not silent:
            # print(the simulation command line output
            print(' ' + '-'*75)
            print(''.join(cases[case]['sim_STDOUT']))
            print(' ' + '-'*75)
            # caclulation time
            stp = time() - start
            stpmin = stp/60.
            print('HAWC2 execution time: %8.2f sec (%8.2f min)' % (stp,stpmin))

        # where there any errors in the output? If yes, abort
        for k in cases[case]['sim_STDOUT']:
            kstart = k[:14]
            if kstart in [' *** ERROR ***', 'forrtl: severe']:
                cases[case]['[hawc2_sim_ok]'] = False
                #raise UserWarning, 'Found error in HAWC2 STDOUT'
            else:
                cases[case]['[hawc2_sim_ok]'] = True

        # check the log file strait away if required
        if check_log:
            start = time()
            errorlogs = logcheck_case(errorlogs, cases, case, silent=silent)
            stop = time() - start
            if case.endswith('.htc'):
                kk = case[:-4] + '.log'
            else:
                kk = case + '.log'
            errors = errorlogs.MsgListLog2[kk][0]
            exitok = errorlogs.MsgListLog2[kk][1]
            if not silent:
                print('log checks took %5.2f sec' % stop)
                print('    found error: ', errors)
                print(' exit correctly: ', exitok)
                print('*'*75)
                print()
            # also save in cases
            if not errors and exitok:
                cases[case]['[hawc2_sim_ok]'] = True
            else:
                cases[case]['[hawc2_sim_ok]'] = False

    if check_log:
        # take the last case to determine sim_id, run_dir and log_dir
        sim_id = cases[case]['[sim_id]']
        run_dir = cases[case]['[run_dir]']
        log_dir = cases[case]['[log_dir]']
        # save the extended (.csv format) errorlog list?
        # but put in one level up, so in the logfiles folder directly
        errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
        # use the model path of the last encoutered case in cases
        errorlogs.PathToLogs = os.path.join(run_dir, log_dir)
        errorlogs.save()

    # just in case, browse back the working path relevant for the python magic
    os.chdir(cwd)
    if not silent:
        print('\nHAWC2 has done all of its sequential magic!')
        print('='*79)
        print('')

    return cases

def prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                write_htc=True, runmethod='none', verbose=False,
                copyback_turb=True, msg='', silent=False, check_log=True,
                update_cases=False, ignore_non_unique=False,
                run_only_new=False, windows_nr_cpus=2,
                pbs_fname_appendix=True, short_job_names=True, qsub='',
                update_model_data=False, maxcpu=1, pyenv='py36-wetb',
                m=[3,4,6,8,9,10,12], postpro_node_zipchunks=True,
                postpro_node=False, exesingle=None, exechunks=None,
                wine_arch='win32', wine_prefix='~/.wine32', prelude='',
                pyenv_cmd='source /home/python/miniconda3/bin/activate'):
    """
    Create the htc files, pbs scripts and replace the tags in master file
    =====================================================================

    Do not use any uppercase letters in the filenames, since HAWC2 will
    convert all of them to lower case results file names (.sel, .dat, .log)

    create sub folders according to sim_id, in order to not create one
    folder for the htc, results, logfiles which grows very large in due
    time!!

    opt_tags is a list of dictionaries of tags:
        [ {tag1=12,tag2=23,..},{tag1=11, tag2=33, tag9=5,...},...]
    for each wind, yaw and coning combi, each tag dictionary in the list
    will be set.

    Make sure to always define all dictionary keys in each list, otherwise
    the value of the first appareance will remain set for the remaining
    simulations in the list.
    For instance, in the example above, if tag9=5 is not set for subsequent
    lists, tag9 will remain having value 5 for these subsequent sets

    The tags for each case are consequently set in following order (or
    presedence):
        * master
        * opt_tags
        * iter_dict
        * variable_tag_func

    Parameters
    ----------

    iter_dict : dict

    opt_tags : list

    master : HtcMaster object

    variable_tag_func : function object

    write_htc : boolean, default=True

    verbose : boolean, default=False

    runmethod : {'none' (default),'pbs','linux-script','local',
                 'local-ram', 'windows-script'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'linux-script' a shell
        script is written to run all cases locally sequential. If set to
        'pbs', PBS scripts are written for a cluster (e.g. Gorm/jess).
        A Windows batch script is written in case of windows-script, and is
        used in combination with windows_nr_cpus.

    msg : str, default=''
        A descriptive message of the simulation series is saved at
        "post_dir + master.tags['[sim_id]'] + '_tags.txt'". Additionally, this
         tagfile also holds the opt_tags and iter_dict values.

    update_cases : boolean, default=False
        If True, a current cases dictionary can be updated with new simulations

    qsub : str, default=''
        Valid options are 'time' (use with launch), 'depend' (use with launch.py
        --depend) or '' (use with launch.py).
        Empty string means there are no tags placed in the pbs file, and
        consequently the pbs file can be submitted as is. When using
        qsub='time', a start time option is inserted with a start time tag
        that has to be set at launch time. With 'depend', a job_id dependency
        line is added, and when launching the job this dependency needs to
        specified.

    update_model_data : default=False
        If set to False, the zip file will not be created, and the data files
        are not copied to the run_dir.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    """

    post_dir = master.tags['[post_dir]']
    fpath_post_base = os.path.join(post_dir, master.tags['[sim_id]'])
    # either take a currently existing cases dictionary, or create a new one
    if update_cases:
        try:
            FILE = open(fpath_post_base + '.pkl', 'rb')
            cases = pickle.load(FILE)
            FILE.close()
            print('updating cases for %s' % master.tags['[sim_id]'])
        except IOError:
            print(79*'=')
            print("failed to load cases dict for updating simd_id at:")
            print(fpath_post_base + '.pkl')
            print(79*'=')
            cases = {}
        # but only run the new cases
        cases_to_run = {}
    else:
        cases = {}

    # if empty, just create a dummy item so we get into the loops
    if len(iter_dict) == 0:
        iter_dict = {'__dummy__': [0]}
    combi_list = create_multiloop_list(iter_dict)

    # load the master htc file as a string under the master.tags
    master.loadmaster()
    # save a copy of the default values
    mastertags_default = copy.copy(master.tags)

    # ignore if the opt_tags is empty, will result in zero
    if len(opt_tags) > 0:
        sim_total = len(combi_list)*len(opt_tags)
    else:
        sim_total = len(combi_list)
        # if no opt_tags specified, create an empty dummy tag
        opt_tags = [dict({'__DUMMY_TAG__' : 0})]
    sim_nr = 0

    # make sure all the required directories are in place at run_dir
#    master.create_run_dir()
#    master.init_multithreads()

    # cycle thourgh all the combinations
    for it in combi_list:
        for ot in opt_tags:
            sim_nr += 1
            # starting point should always be the default values. This is
            # important when a previous case had a certain tag defined, and in
            # the next case it is absent.
            master.tags = mastertags_default.copy()
            # update the tags from the opt_tags list
            if not '__DUMMY_TAG__' in ot:
                master.tags.update(ot)
            # update the tags set in the combi_list
            master.tags.update(it)
            # force lower case values as defined in output_dirs
            master.lower_case_output()
            # -----------------------------------------------------------
            # start variable tags update
            if variable_tag_func is not None:
                master = variable_tag_func(master)
            # end variable tags
            # -----------------------------------------------------------
            if not silent:
                print('htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                       format(sim_total, '3.0f'))

            if verbose:
                print('===master.tags===\n', master.tags)

            # returns a dictionary with all the tags used for this
            # specific case
            htc = master.createcase(write_htc=write_htc)
            if update_model_data:
                master.create_run_dir()
            #htc=master.createcase_check(cases_repo,write_htc=write_htc)

            # make sure the current cases is unique!
            if not ignore_non_unique:
                if list(htc.keys())[0] in cases:
                    msg = 'non unique case in cases: %s' % list(htc.keys())[0]
                    raise KeyError(msg)

            # save in the big cases. Note that values() gives a copy!
            cases[list(htc.keys())[0]] = list(htc.values())[0]
            # if we have an update scenario, keep track of the cases we want
            # to run again. This prevents us from running all cases on every
            # update
            if run_only_new:
                cases_to_run[list(htc.keys())[0]] = list(htc.values())[0]

            if verbose:
                print('created cases for: %s.htc\n' % master.tags['[case_id]'])

#    print(master.queue.get())

    # only copy data and create zip after all htc files have been created.
    # Note that createcase could also creat other input files
    # create the execution folder structure and copy all data to it
    # FIXME: this approach only considers the tags as set in the last case!
    if update_model_data:
        master.copy_model_data()
        # create the zip file
        master.create_model_zip()

    # create directory if post_dir does not exists
    try:
        os.makedirs(post_dir)
    except OSError:
        pass
    FILE = open(fpath_post_base + '.pkl', 'wb')
    pickle.dump(cases, FILE, protocol=2)
    FILE.close()

    if not silent:
        print('\ncases saved at:')
        print(fpath_post_base + '.pkl')

    # also save the iter_dict and opt_tags in a text file for easy reference
    # or quick checks on what each sim_id actually contains
    # sort the taglist for convienent reading/comparing
    tagfile = msg + '\n\n'
    tagfile += '='*79 + '\n'
    tagfile += 'iter_dict\n'.rjust(30)
    tagfile += '='*79 + '\n'
    iter_dict_list = sorted(iter(iter_dict.items()), key=itemgetter(0))
    for k in iter_dict_list:
        tagfile += str(k[0]).rjust(30) + ' : ' + str(k[1]).ljust(20) + '\n'

    tagfile += '\n'
    tagfile += '='*79 + '\n'
    tagfile += 'opt_tags\n'.rjust(30)
    tagfile += '='*79 + '\n'
    for k in opt_tags:
        tagfile += '\n'
        tagfile += '-'*79 + '\n'
        tagfile += 'opt_tags set\n'.rjust(30)
        tagfile += '-'*79 + '\n'
        opt_dict = sorted(iter(k.items()), key=itemgetter(0), reverse=False)
        for kk in opt_dict:
            tagfile += str(kk[0]).rjust(30)+' : '+str(kk[1]).ljust(20) + '\n'
    if update_cases:
        mode = 'a'
    else:
        mode = 'w'
    write_file(fpath_post_base + '_tags.txt', tagfile, mode)

    if run_only_new:
        cases = cases_to_run

    launch(cases, runmethod=runmethod, verbose=verbose, check_log=check_log,
           copyback_turb=copyback_turb, qsub=qsub,
           windows_nr_cpus=windows_nr_cpus, short_job_names=short_job_names,
           pbs_fname_appendix=pbs_fname_appendix, silent=silent, maxcpu=maxcpu,
           pyenv=pyenv, m=[3,4,6,8,9,10,12],
           postpro_node_zipchunks=postpro_node_zipchunks, prelude=prelude,
           postpro_node=postpro_node, exesingle=exesingle, exechunks=exechunks,
           wine_arch=wine_arch, wine_prefix=wine_prefix, pyenv_cmd=pyenv_cmd)

    return cases

def launch(cases, runmethod='none', verbose=False, copyback_turb=True,
           silent=False, check_log=True, windows_nr_cpus=2, qsub='time',
           pbs_fname_appendix=True, short_job_names=True,
           maxcpu=1, pyenv='py36-wetb', m=[3,4,6,8,9,10,12], prelude='',
           postpro_node_zipchunks=True, postpro_node=False, exesingle=None,
           exechunks=None, wine_arch='win32', wine_prefix='~/.wine32',
           pyenv_cmd='source /home/python/miniconda3/bin/activate'):
    """
    The actual launching of all cases in the Cases dictionary. Note that here
    only the PBS files are written and not the actuall htc files.

    Parameters
    ----------

    cases : dict
        Dictionary with the case name as key and another dictionary as value.
        The latter holds all the tag/value pairs used in the respective
        simulation.

    verbose : boolean, default=False

    runmethod : {'none' (default),'pbs','linux-script','local',
                 'local-ram', 'windows-script'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'linux-script' a shell
        script is written to run all cases locally sequential. If set to
        'pbs', PBS scripts are written for a cluster (e.g. Gorm/jess).
        A Windows batch script is written in case of windows-script, and is
        used in combination with windows_nr_cpus.

    windows_nr_cpus : int, default=2
        All cases to be run are distributed over 'windows_nr_cpus' number of
        Windows batch files so the user can utilize 'windows_nr_cpus' CPUs.
    """

    random_case = list(cases.keys())[0]
    sim_id = cases[random_case]['[sim_id]']
    pbs_out_dir = cases[random_case]['[pbs_out_dir]']

    if runmethod == 'local-script' or runmethod == 'linux-script':
        local_shell_script(cases, sim_id)
    elif runmethod == 'windows-script':
        local_windows_script(cases, sim_id, nr_cpus=windows_nr_cpus)
    elif runmethod in ['pbs','jess','gorm']:
        # create the pbs object
        pbs = PBS(cases, short_job_names=short_job_names, pyenv=pyenv,
                  pbs_fname_appendix=pbs_fname_appendix, qsub=qsub,
                  verbose=verbose, silent=silent, prelude=prelude,
                  m=m, postpro_node_zipchunks=postpro_node_zipchunks,
                  postpro_node=postpro_node, exesingle=exesingle,
                  exechunks=exechunks, wine_arch=wine_arch,
                  wine_prefix=wine_prefix)
        pbs.pyenv_cmd = pyenv_cmd
        pbs.copyback_turb = copyback_turb
        pbs.pbs_out_dir = pbs_out_dir
        pbs.maxcpu = maxcpu
        pbs.create()
    elif runmethod == 'local':
        cases = run_local(cases, silent=silent, check_log=check_log)
    elif runmethod =='local-ram':
        cases = run_local_ram(cases, check_log=check_log)
    elif runmethod == 'none':
        pass
    else:
        msg = 'unsupported runmethod, valid options: local, linux-script, ' \
              'windows-script, local-ram, none, pbs'
        raise ValueError(msg)

def post_launch(cases, save_iter=False, silent=False, suffix=None,
                path_errorlog=None):
    """
    Do some basics checks: do all launched cases have a result and LOG file
    and are there any errors in the LOG files?

    Parameters
    ----------

    cases : either a string (path to file) or the cases itself

    save_iter : boolean, default=False
        Set to True to save the number of iterations per time step in
        *.iter file (in the same folder as the logfile)

    path_errorlog : str, default=None
        Root path of the error logfiles. If set to None (default), the
        value set in the [run_dir] tag is used as the root folder of the
        logfiles.

    suffix : str, default=None
        If not None, the suffix will be appended to file name of the error
        log analysis file as follows: "ErrorLog_suffix.csv".
    """

    # TODO: finish support for default location of the cases and file name
    # two scenario's: either pass on an cases and get from their the
    # post processing path or pass on the simid and load from the cases
    # from the default location
    # in case run_local, do not check PBS!

    # in case it is a path, load the cases
    if type(cases).__name__ == 'str':
        cases = load_pickled_file(cases)

    # saving output to textfile and print(at the same time
    LOG = Log()
    LOG.print_logging = True

    # load one case dictionary from the cases to get data that is the same
    # over all simulations in the cases
    try:
        master = list(cases.keys())[0]
    except IndexError:
        print('there are no cases, aborting...')
        return None
    post_dir = cases[master]['[post_dir]']
    sim_id = cases[master]['[sim_id]']
    run_dir = cases[master]['[run_dir]']
    log_dir = cases[master]['[log_dir]']

    # for how many of the created cases are there actually result, log files
    pbs = PBS(cases)
    pbs.cases = cases
    cases_fail = pbs.check_results(cases)

    # add the failed cases to the LOG:
    LOG.add(['number of failed cases: ' + str(len(cases_fail))])
    LOG.add(list(cases_fail))
    # for k in cases_fail:
    #    print(k

    # initiate the object to check the log files
    errorlogs = ErrorLogs(cases=cases)
    LOG.add(['checking ' + str(len(cases)) + ' LOG files...'])
    nr = 1
    nr_tot = len(cases)

    tmp = list(cases.keys())[0]
    if not silent:
        print('checking logs, path (from a random item in cases):')
        print(os.path.join(run_dir, log_dir))

    for k in sorted(cases.keys()):
        # a case could not have a result, but a log file might still exist
        if k.endswith('.htc'):
            kk = k[:-4] + '.log'
        else:
            kk = k + '.log'
        # note that if errorlogs.PathToLogs is a file, it will only check that
        # file. If it is a directory, it will check all that is in the dir
        run_dir = cases[k]['[run_dir]']
        log_dir = cases[k]['[log_dir]']
        errorlogs.PathToLogs = os.path.join(run_dir, log_dir, kk)
        try:
            errorlogs.check(save_iter=save_iter)
            if not silent:
                print('checking logfile progress: % 6i/% 6i' % (nr, nr_tot))
        except IOError:
            if not silent:
                print('           no logfile for:  %s' % (errorlogs.PathToLogs))
        except Exception as e:
            if not silent:
                print('  log analysis failed for: %s' % kk)
                print(e)
        nr += 1

        # if simulation did not ended correctly, put it on the fail list
        try:
            if not errorlogs.MsgListLog2[kk][1]:
                cases_fail[k] = cases[k]
        except KeyError:
            pass

    # now see how many cases resulted in an error and add to the general LOG
    # determine how long the first case name is
    try:
        spacing = len(list(errorlogs.MsgListLog2.keys())[0]) + 9
    except Exception as e:
        print('nr of OK cases: %i' % (len(cases) - len(cases_fail)))
        raise(e)
    LOG.add(['display log check'.ljust(spacing) + 'found_error?'.ljust(15) + \
            'exit_correctly?'])
    for k in errorlogs.MsgListLog2:
        LOG.add([k.ljust(spacing)+str(errorlogs.MsgListLog2[k][0]).ljust(15)+\
            str(errorlogs.MsgListLog2[k][1]) ])
    # save the extended (.csv format) errorlog list?
    # but put in one level up, so in the logfiles folder directly
    errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    # save the log file analysis in the run_dir instead of the log_dir
    if path_errorlog is None:
        errorlogs.PathToLogs = run_dir# + log_dir
    else:
        errorlogs.PathToLogs = path_errorlog
    errorlogs.save(suffix=suffix)

    # save the error LOG list, this is redundant, since it already exists in
    # the general LOG file (but only as a print, not the python variable)
    tmp = os.path.join(post_dir, sim_id + '_MsgListLog2')
    save_pickle(tmp, errorlogs.MsgListLog2)

    # save the list of failed cases
    save_pickle(os.path.join(post_dir, sim_id + '_fail.pkl'), cases_fail)

    return cases_fail

def copy_pbs_in_failedcases(cases_fail, path='pbs_in_fail', silent=True):
    """
    Copy all the pbs_in files from failed cases to a new directory so it
    is easy to re-launch them
    """
    if not silent:
        print('Following failed cases pbs_in files are copied:')
    for cname in cases_fail.keys():
        case = cases_fail[cname]
        pbs_in_fname = '%s.p' % (case['[case_id]'])
        run_dir = case['[run_dir]']

        src = os.path.join(run_dir, case['[pbs_in_dir]'], pbs_in_fname)

        pbs_in_dir_fail = case['[pbs_in_dir]'].replace('pbs_in', path)
        dst = os.path.join(run_dir, pbs_in_dir_fail, pbs_in_fname)

        if not silent:
            print(dst)
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        shutil.copy2(src, dst)

def logcheck_case(errorlogs, cases, case, silent=False):
    """
    Check logfile of a single case
    ==============================

    Given the cases and a case, check that single case on errors in the
    logfile.

    """

    #post_dir = cases[case]['[post_dir]']
    #sim_id = cases[case]['[sim_id]']
    run_dir = cases[case]['[run_dir]']
    log_dir = cases[case]['[log_dir]']
    if case.endswith('.htc'):
        caselog = case[:-4] + '.log'
    else:
        caselog = case + '.log'
    errorlogs.PathToLogs = os.path.join(run_dir, log_dir, caselog)
    errorlogs.check()

    # in case we find an error, abort or not?
    errors = errorlogs.MsgListLog2[caselog][0]
    exitcorrect = errorlogs.MsgListLog2[caselog][1]
    if errors:
        # print all error messages
        #logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], ]
        # difficult: MsgListLog is not a dict!!
        #raise UserWarning, 'HAWC2 simulation has errors in logfile, abort!'
        #warnings.warn('HAWC2 simulation has errors in logfile!')
        logging.warn('HAWC2 simulation has errors in logfile!')
    elif not exitcorrect:
        #raise UserWarning, 'HAWC2 simulation did not ended correctly, abort!'
        #warnings.warn('HAWC2 simulation did not ended correctly!')
        logging.warn('HAWC2 simulation did not ended correctly!')

    # no need to do that, aborts on failure anyway and OK log check will be
    # printed in run_local when also printing how long it took to check
    #if not silent:
        #print 'log checks ok'
        #print '   found error: %s' % errorlogs.MsgListLog2[caselog][0]
        #print 'exit correctly: %s' % errorlogs.MsgListLog2[caselog][1]

    return errorlogs

    ## save the extended (.csv format) errorlog list?
    ## but put in one level up, so in the logfiles folder directly
    #errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    ## use the model path of the last encoutered case in cases
    #errorlogs.PathToLogs = run_dir + log_dir
    #errorlogs.save()


class Log(object):
    """
    Class for convinient logging. Create an instance and add lines to the
    logfile as a list with the function add.
    The added items will be printed if
        self.print_logging = True. Default value is False

    Create the instance, add with .add('lines') (lines=list), save with
    .save(target), print(current log to screen with .printLog()
    """
    def __init__(self):
        self.log = []
        # option, should the lines added to the log be printed as well?
        self.print_logging = False
        self.file_mode = 'a'

    def add(self, lines):
        # the input is a list, where each entry is considered as a new line
        for k in lines:
            self.log.append(k)
            if self.print_logging:
                print(k)

    def save(self, target):
        # tread every item in the log list as a new line
        FILE = open(target, self.file_mode)
        for k in self.log:
            FILE.write(k + '\n')
        FILE.close()
        # and empty the log again
        self.log = []

    def printscreen(self):
        for k in self.log:
            print(k)


class HtcMaster(object):
    """
    """

    def __init__(self, verbose=False, silent=False):
        """
        """

        # TODO: make HtcMaster callable, so that when called you actually
        # set a value for a certain tag or add a new one. In doing so,
        # you can actually warn when you are overwriting a tag, or when
        # a different tag has the same name, etc

        # create a dictionary with the tag name as key as the default value
        self.tags = dict()

        # should we print(where the file is written?
        self.verbose = verbose
        self.silent = silent

        # following tags are required
        #---------------------------------------------------------------------
        self.tags['[case_id]'] = None

        self.tags['[master_htc_file]'] = None
        self.tags['[master_htc_dir]'] = None
        # path to model zip file, needs to accessible from the server
        # relative from the directory where the pbs files are launched on the
        # server. Suggestions is to always place the zip file in the model
        # folder, so only the zip file name has to be defined
        self.tags['[model_zip]'] = None

        # path to HAWTOPT blade result file: quasi/res/blade.dat
        self.tags['[blade_hawtopt_dir]'] = None
        self.tags['[blade_hawtopt]'] = None
        self.tags['[zaxis_fact]'] = 1.0
        # TODO: rename to execution dir, that description fits much better!
        self.tags['[run_dir]'] = None
        #self.tags['[run_dir]'] = '/home/dave/tmp/'

        # following dirs are relative to the run_dir!!
        # they indicate the location of the SAVED (!!) results, they can be
        # different from the execution dirs on the node which are set in PBS
        self.tags['[hawc2_exe]'] = 'hawc2mb.exe'
        self.tags['[data_dir]'] = 'data/'
        self.tags['[res_dir]'] = 'results/'
        self.tags['[iter_dir]'] = 'iter/'
        self.tags['[log_dir]'] = 'logfiles/'
        self.tags['[turb_dir]'] = 'turb/'
        self.tags['[micro_dir]'] = None
        self.tags['[meander_dir]'] = None
        self.tags['[turb_db_dir]'] = None
        self.tags['[micro_db_dir]'] = None
        self.tags['[meander_db_dir]'] = None
        self.tags['[control_dir]'] = 'control/'
        self.tags['[externalforce]'] = 'externalforce/'
        self.tags['[animation_dir]'] = 'animation/'
        self.tags['[eigenfreq_dir]'] = 'eigenfreq/'
        self.tags['[htc_dir]'] = 'htc/'
        self.tags['[mooring_dir]'] = 'mooring/'
        self.tags['[hydro_dir]'] = 'htc_hydro/'
        self.tags['[pbs_out_dir]'] = 'pbs_out/'
        self.tags['[turb_base_name]'] = None
        self.tags['[micro_base_name]'] = None
        self.tags['[meander_base_name]'] = None
        self.tags['[zip_root_files]'] = []

        self.tags['[fname_source]'] = []
        self.tags['[fname_default_target]'] = []

        self.tags['[eigen_analysis]'] = False

        # the express que has 2 thyra nodes with max walltime of 1h
#        self.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
        # walltime should have following format: hh:mm:ss
        self.tags['[walltime]'] = '04:00:00'

#        self.queue = Queue.Queue()

        self.output_dirs = ['[res_dir]', '[log_dir]', '[turb_dir]',
                            '[case_id]', '[micro_dir]', '[animation_dir]',
                            '[meander_dir]', '[eigenfreq_dir]']

    def create_run_dir(self):
        """
        If non existent, create run_dir and all required model sub directories
        """

        dirkeys = ['[data_dir]', '[htc_dir]', '[res_dir]', '[log_dir]',
                   '[eigenfreq_dir]', '[animation_dir]', '[turb_dir]',
                   '[micro_dir]', '[meander_dir]', '[opt_dir]', '[control_dir]',
                   '[mooring_dir]', '[hydro_dir]', '[externalforce]']

        # create all the necessary directories
        for dirkey in dirkeys:
            if isinstance(self.tags[dirkey], str):
                path = os.path.join(self.tags['[run_dir]'], self.tags[dirkey])
                if self.tags[dirkey].lower() == 'none':
                    continue
                if not os.path.exists(path):
                    os.makedirs(path)

    # TODO: copy_model_data and create_model_zip should be the same.
    def copy_model_data(self):
        """

        Copy the model data to the execution folder

        """

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the data files, they are already on location
        data_local = os.path.join(self.tags['[model_dir_local]'],
                                  self.tags['[data_dir]'])
        data_run = os.path.join(self.tags['[run_dir]'], self.tags['[data_dir]'])
        if data_local == data_run:
            return

        # copy root files
        model_root = self.tags['[model_dir_local]']
        run_root = self.tags['[run_dir]']
        for fname in self.tags['[zip_root_files]']:
            shutil.copy2(model_root + fname, run_root + fname)

        # copy special files with changing file names
        if '[ESYSMooring_init_fname]' in self.tags:
            if isinstance(self.tags['[ESYSMooring_init_fname]'], str):
                fname_source = self.tags['[ESYSMooring_init_fname]']
                fname_target = 'ESYSMooring_init.dat'
                shutil.copy2(model_root + fname_source,
                             run_root + fname_target)

        # copy the master file into the htc/_master dir
        src = os.path.join(self.tags['[master_htc_dir]'],
                           self.tags['[master_htc_file]'])
        # FIXME: htc_dir can contain the DLC folder name
        dst = os.path.join(self.tags['[run_dir]'], 'htc', '_master')
        if not os.path.exists(dst):
            os.makedirs(dst)
        shutil.copy2(src, dst)

        # copy all content of the following dirs
        dirs = [self.tags['[control_dir]'], self.tags['[hydro_dir]'],
                self.tags['[mooring_dir]'], self.tags['[externalforce]'],
                self.tags['[data_dir]'], 'htc/DLCs/']
        plocal = self.tags['[model_dir_local]']
        prun = self.tags['[run_dir]']

        # copy all files present in the specified folders
        for path in dirs:
            if not path:
                continue
            elif not os.path.exists(os.path.join(plocal, path)):
                continue
            for root, dirs, files in os.walk(os.path.join(plocal, path)):
                for file_name in files:
                    src = os.path.join(root, file_name)
                    dst = os.path.abspath(root).replace(os.path.abspath(plocal),
                                       os.path.abspath(prun))
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    dst = os.path.join(dst, file_name)
                    shutil.copy2(src, dst)

        # and last copies: the files with generic input names
        if not isinstance(self.tags['[fname_source]'], list):
            raise ValueError('[fname_source] needs to be a list')
        if not isinstance(self.tags['[fname_default_target]'], list):
            raise ValueError('[fname_default_target] needs to be a list')
        len1 = len(self.tags['[fname_source]'])
        len2 = len(self.tags['[fname_default_target]'])
        if len1 != len2:
            raise ValueError('[fname_source] and [fname_default_target] '
                             'need to have the same number of items')
        for i in range(len1):
            src = os.path.join(plocal, self.tags['[fname_source]'][i])
            dst = os.path.join(prun, self.tags['[fname_default_target]'][i])
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            shutil.copy2(src, dst)

    # TODO: copy_model_data and create_model_zip should be the same.
    def create_model_zip(self):
        """

        Create the model zip file based on the master tags file settings.

        Paremeters
        ----------

        master : HtcMaster object


        """

        # FIXME: all directories should be called trough their appropriate tag!

        #model_dir = HOME_DIR + 'PhD/Projects/Hawc2Models/'+MODEL+'/'
        model_dir_server = self.tags['[run_dir]']
        model_dir_local = self.tags['[model_dir_local]']

        # ---------------------------------------------------------------------
        # create the zipfile object locally
        fname = os.path.join(model_dir_local, self.tags['[model_zip]'])
        zf = zipfile.ZipFile(fname, 'w')

        # empty folders, the'll hold the outputs
        # zf.write(source, target in zip, )
        # TODO: use user defined directories here and in PBS
        # note that they need to be same as defined in the PBS script. We
        # manually set these up instead of just copying the original.

#        animation_dir = self.tags['[animation_dir]']
#        eigenfreq_dir = self.tags['[eigenfreq_dir]']
#        logfiles_dir = self.tags['[log_dir]']
#        results_dir = self.tags['[res_dir]']
#        htc_dir = self.tags['[htc_dir]']
        htcmaster = self.tags['[master_htc_file]']

        control_dir = self.tags['[control_dir]']
        htcmaster_dir = self.tags['[master_htc_dir]']
        data_dir = self.tags['[data_dir]']
        turb_dir = self.tags['[turb_dir]']
        wake_dir = self.tags['[micro_dir]']
        meander_dir = self.tags['[meander_dir]']
        mooring_dir = self.tags['[mooring_dir]']
        hydro_dir = self.tags['[hydro_dir]']
        extforce = self.tags['[externalforce]']
        # result dirs are not required, HAWC2 will create them
        dirs = [control_dir, data_dir, extforce, turb_dir, wake_dir,
                meander_dir, mooring_dir, hydro_dir]
        for zipdir in dirs:
            if zipdir:
                zf.write('.', os.path.join(zipdir, '.'), zipfile.ZIP_DEFLATED)
        zf.write('.', 'htc/_master/.', zipfile.ZIP_DEFLATED)

        # if any, add files that should be added to the root of the zip file
        for file_name in self.tags['[zip_root_files]']:
            src = os.path.join(model_dir_local, file_name)
            zf.write(src, file_name, zipfile.ZIP_DEFLATED)

        if '[ESYSMooring_init_fname]' in self.tags:
            if self.tags['[ESYSMooring_init_fname]'] is not None:
                fname_source = self.tags['[ESYSMooring_init_fname]']
                fname_target = 'ESYSMooring_init.dat'
                zf.write(model_dir_local + fname_source, fname_target,
                         zipfile.ZIP_DEFLATED)

        # the master file
        src = os.path.join(htcmaster_dir, htcmaster)
        dst = os.path.join('htc', '_master', os.path.basename(htcmaster))
        zf.write(src, dst, zipfile.ZIP_DEFLATED)

        # manually add all that resides in control, mooring and hydro
        paths = [control_dir, mooring_dir, hydro_dir, extforce, data_dir]
        for target_path in paths:
            if not target_path:
                continue
            path_src = os.path.join(model_dir_local, target_path)
            for root, dirs, files in os.walk(path_src):
                for file_name in files:
                    #print 'adding', file_name
                    src = os.path.join(root, file_name)
                    # the zip file only contains the relative paths
                    rel_dst = root.replace(os.path.abspath(model_dir_local), '')
                    if os.path.isabs(rel_dst):
                        rel_dst = rel_dst[1:]
                    rel_dst = os.path.join(rel_dst, file_name)
                    zf.write(src, rel_dst, zipfile.ZIP_DEFLATED)

        # and last copies: the files with generic input names
        if not isinstance(self.tags['[fname_source]'], list):
            raise ValueError('[fname_source] needs to be a list')
        if not isinstance(self.tags['[fname_default_target]'], list):
            raise ValueError('[fname_default_target] needs to be a list')
        len1 = len(self.tags['[fname_source]'])
        len2 = len(self.tags['[fname_default_target]'])
        if len1 != len2:
            raise ValueError('[fname_source] and [fname_default_target] '
                             'need to have the same number of items')
        for i in range(len1):
            src = os.path.join(model_dir_local, self.tags['[fname_source]'][i])
            # the zip file only contains the relative paths
            rel_dst = self.tags['[fname_default_target]'][i]
            # we can not have an absolute path here, make sure it isn't
            if os.path.isabs(rel_dst):
                rel_dst = rel_dst[1:]
            zf.write(src, rel_dst, zipfile.ZIP_DEFLATED)

        # and close again
        zf.close()

        # ---------------------------------------------------------------------
        # copy zip file to the server, this will be used on the nodes
        src = os.path.join(model_dir_local, self.tags['[model_zip]'])
        dst = os.path.join(model_dir_server, self.tags['[model_zip]'])

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the zip file, it is already on location
        if not src == dst:
            shutil.copy2(src, dst)

        ## copy to zip data file to sim_id htc folder on the server dir
        ## so we now have exactly all data to relaunch any htc file later
        #dst  = model_dir_server + self.tags['[htc_dir]']
        #dst += self.tags['[model_zip]']
        #shutil.copy2(src, dst)

    def loadmaster(self):
        """
        Load the master file, path to master file is defined in
        __init__(): target, master. Additionally, find all the tags in the
        master file. Note that tags [] in the label and comment sections are
        ignored.

        All the tags that are found in the master file are saved in the
        self.tags_in_master dictionary, with the line numbers in a list as
        values:
        tags_in_master[tagname] = [line nr occurance 1, line nr occurance 2, ]
        note that tagname includes the []
        """

        # what is faster, load the file in one string and do replace()?
        # or the check error log approach?
        fpath  = os.path.join(self.tags['[master_htc_dir]'],
                              self.tags['[master_htc_file]'])
        # load the file:
        if not self.silent:
            print('loading master: ' + fpath)

        lines = misc.readlines_try_encodings(fpath)

        # regex for finding all tags in a line
        regex = re.compile('(\\[.*?\\])')
        self.tags_in_master = {}

        # convert to string:
        self.master_str = ''
        for i, line in enumerate(lines):
            # are there any tags on this line? Ignore comment AND label section
            tags = regex.findall(line.split(';')[0].split('#')[0])
            for tag in tags:
                try:
                    self.tags_in_master[tag].append(i)
                except KeyError:
                    self.tags_in_master[tag] = [i]
            # safe for later
            self.master_str += line

    def createcase_check(self, htc_dict_repo, \
                            tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        Check if a certain case name already exists in a specified htc_dict.
        If true, return a message and do not create the case. It can be that
        either the case name is a duplicate and should be named differently,
        or that the simulation is a duplicate and it shouldn't be repeated.
        """

        # is the [case_id] tag unique, given the htc_dict_repo?
        if self.verbose:
            print('checking if following case is in htc_dict_repo: ')
            print(self.tags['[case_id]'] + '.htc')

        if self.tags['[case_id]'] + '.htc' in htc_dict_repo:
            # if the new case_id already exists in the htc_dict_repo
            # do not add it again!
            # print('case_id key is not unique in the given htc_dict_repo!'
            raise UserWarning('case_id key is not unique in the given '
                              'htc_dict_repo!')
        else:
            htc = self.createcase(tmp_dir=tmp_dir, write_htc=write_htc)
            return htc

    def createcase(self, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        replace all the tags from the master file and save the new htc file
        """

        htc = self.master_str

        # FIXME: HAWC2 always outputs result and logfile in lower case, so
        # force case_id/Case id. to be lower case
        self.tags['[case_id]'] = self.tags['[case_id]'].lower()
        if '[Case id.]' in self.tags:
            self.tags['[Case id.]'] = self.tags['[Case id.]'].lower()

        # and now replace all the tags in the htc master file
        # when iterating over a dict, it will give the key, given in the
        # corresponding format (string keys as strings, int keys as ints...)
        for k in self.tags:
            # TODO: give error if a default is not defined, like null
            # if it is a boolean, replace with ; or blank
            if isinstance(self.tags[k], bool):
                if self.tags[k]:
                    # we have a boolean that is True, switch it on
                    value = ''
                else:
                    value = ';'
            else:
                value = self.tags[k]
            # if string is not found, it will do nothing
            # case sensitive tag names
#            htc = htc.replace(str(k), str(value))
            # tag names are case insensitive
            htc = re.sub(re.escape(str(k)), str(value), htc, flags=re.IGNORECASE)

        # and save the the case htc file:
        cname = self.tags['[case_id]'] + '.htc'

        htc_target = os.path.join(self.tags['[run_dir]'], self.tags['[htc_dir]'])
        if not self.silent:
            print('htc will be written to: ')
            print('  ' + htc_target)
            print('  ' + cname)

        # and write the htc file to the temp dir first
        if write_htc:
            self.write_htc(cname, htc, htc_target)
#            thread = threading.Thread(target=self.write_htc,
#                                      args=(cname, htc, htc_target))
#            thread.daemon = True
#            thread.start()
        # save all the tags for debugging purpuses
        if self.verbose:
            tmp = ''
            for key in sorted(self.tags.keys()):
                value = self.tags[key]
                rpl = (key.rjust(25), str(value).ljust(70),
                       type(key).__name__.ljust(10), type(value).__name__)
                tmp += '%s -- %s -- %s -- %s\n' % rpl
            write_file(htc_target + cname + '.tags', tmp, 'w')

        # return the used tags, some parameters can be used later, such as the
        # turbulence name in the pbs script
        # return as a dictionary, to be used in htc_dict
        # return a copy of the tags, otherwise you will not catch changes
        # made to the different tags in your sim series
        return {cname : copy.copy(self.tags)}

    def write_htc(self, cname, htc, htc_target):
        # create subfolder if necesarry
        if not os.path.exists(htc_target):
            os.makedirs(htc_target)
        write_file(htc_target + cname, htc, 'w')
        # write_file(tmp_dir + case, htc, 'w')

    def lower_case_output(self):
        """
        force lower case tags on output files since HAWC2 will force them to
        lower case anyway
        """

        for key in self.output_dirs:
            if isinstance(self.tags[key], str):
                self.tags[key] = self.tags[key].lower()

    def write_tags(self, fname=False):
        """Write all tags to a DLC alike spreadsheet
        """

        if not fname:
            a, b = self.tags['[master_htc_dir]'], self.tags['[master_htc_file]']
            fname  = os.path.join(a, b).replace('.htc', '.xlsx')
        df = pd.DataFrame([], columns=self.tags_in_master.keys())
        df.to_excel(fname)


class PBS(object):
    """
    The part where the actual pbs script is writtin in this class (functions
    create(), starting() and ending() ) is based on the MS Excel macro
    written by Torben J. Larsen

    input a list with htc file names, and a dict with the other paths,
    such as the turbulence file and folder, htc folder and others
    """

    def __init__(self, cases, qsub='time', silent=False, pyenv='py36-wetb',
                 pbs_fname_appendix=True, short_job_names=True, verbose=False,
                 m=[3,4,6,8,9,10,12], exesingle=None, prelude='',
                 postpro_node_zipchunks=True, postpro_node=False, queue='workq',
                 exechunks=None, wine_arch='win32', wine_prefix='.wine32'):
        """
        Define the settings here. This should be done outside, but how?
        In a text file, paramters list or first create the object and than set
        the non standard values??

        where htc_dict is a dictionary with
            [key=case name, value=used_tags_dict]

        where tags as outputted by MasterFile (dict with the chosen options)

        For gorm, maxcpu is set to 1, do not change otherwise you might need to
        change the scratch dir handling.

        qsub : str
            time, or depend. For time each job will need to get a start
            time, which will have to be set by replacing [start_time].
            For depend a job dependency chain will have to be established.
            This will be set via [nodeps] and [job_id]
            When none of the above, neither of them is specified, and
            consequently the pbs file can be submitted without replacing any
            tag. Use qsub=None in combination with the launch.Scheduler

        short_job_names : boolean, default=True
            How should the job be named (relevant for the PBS queueing system)?
            When True, it will be named like HAWC2_123456. With False, the
            case_id will be used as job name.

        """
        self.verbose = verbose
        self.silent = silent
        self.pyenv = pyenv
        self.pyenv_cmd = "source /home/python/miniconda3/bin/activate"
        self.postpro_node_zipchunks = postpro_node_zipchunks
        self.postpro_node = postpro_node
        # Prelude is executed just before HAWC2 is
        self.prelude = prelude

        self.m = m

        wine_prefix = misc.sanitize_wine_prefix(wine_prefix)

        # run in 32-bit or 64-bit mode. Note this uses the same assumptions
        # on how to configure wine in toolbox/pbsutils/config-wine-hawc2.sh
        wineparam = (wine_arch, wine_prefix)
        if wine_arch==None or wine_prefix==None:
            self.winebase = "time "
        else:
            self.winebase = "time WINEARCH='%s' WINEPREFIX=\"%s\" " % wineparam
        self.wine = self.winebase + 'wine'
        self.winenumactl = self.winebase + "numactl --physcpubind=$CPU_NR wine"

        # in case you want to redirect stdout to /dev/nul, append as follows:
        # '> /dev/null 2>&1'
        self.exesingle = exesingle
        if exesingle is None:
            self.exesingle = "{wine:} '{hawc2_exe:}' '{fname_htc:}'"
        # in zipchunks mode we will output std out and err to a separate
        # pbs_out file, and that in addition to the pbs_out_zipchunks file
        self.exechunks = exechunks
        if exechunks is None:
            self.exechunks = "({winenumactl:} '{hawc2_exe:}' '{fname_htc:}') "
            self.exechunks += "2>&1 | tee '{fname_pbs_out:}'"

        # TODO: based on a certain host/architecture you can change these
        self.maxcpu = 1
        self.secperiter = 0.012

        # determine at runtime if winefix has to be ran
        self.winefix = ''
        if wine_arch!=None or wine_prefix!=None:
            self.winefix = "  _HOSTNAME_=`hostname`\n"
            self.winefix += "  if [[ ${_HOSTNAME_:0:1} == \"j\" ]] ; then\n"
            self.winefix += "    WINEARCH='%s' WINEPREFIX=\"%s\" winefix\n" % wineparam
            self.winefix += "  fi\n"

        # the output channels comes with a price tag. Each time step
        # will have a penelty depending on the number of output channels

        self.iterperstep = 8.0 # average nr of iterations per time step
        # lead time: account for time losses when starting a simulation,
        # copying the turbulence data, generating the turbulence
        self.tlead = 5.0*60.0

        # use pbs job name as prefix in the pbs file name
        self.pbs_fname_appendix = pbs_fname_appendix
        self.short_job_names = short_job_names
        # pbs job name prefix
        self.pref = 'HAWC2_'
        # the actual script starts empty
        self.pbs = ''

        # /dev/shm should be the RAM of the cluster
#        self.node_run_root = '/dev/shm'
        self.node_run_root = '/scratch'

        # only allow valid queues for jess
        queues = ('windq', 'workq', 'fatq', 'windfatq', 'xpresq', 'xpresfatq')
        if not queue in queues:
            raise ValueError('Invalid queue name: %s' % queue)
        self.pbs_queue_command = '#PBS -q %s' % queue

        self.cases = cases

        # location of the output messages .err and .out created by the node
        self.pbs_out_dir = 'pbs_out/'
        self.pbs_in_dir = 'pbs_in/'

        # for the start number, take hour/minute combo
        d = datetime.datetime.today()
        tmp = int( str(d.hour)+format(d.minute, '02.0f') )*100
        self.pbs_start_number = tmp
        self.qsub = qsub

#        if quemethod == 'time':
#            self.que_jobdeps = False
#        elif type(quemethod).__name__ == 'int':
#            nr_cpus = quemethod
#            self.que_jobdeps = True
#            nr_jobs = len(cases)
#            jobs_per_cpu = int(math.ceil(float(nr_jobs)/float(nr_cpus)))
#            # precalculate all the job ids
#            self.jobid_deps = []
#            self.jobid_list = []
#            count2 = self.pbs_start_number
#            for cpu in range(nr_cpus):
#                self.jobid_list.extend(range(count2, count2+jobs_per_cpu))
#                # the first job to start does not have a dependency
#                self.jobid_deps.append(None)
#                self.jobid_deps.extend(range(count2, count2+jobs_per_cpu-1))
#                count2 += jobs_per_cpu

        self.copyback_turb = True
        self.copyback_fnames = []
        self.copyback_fnames_rename = []
        self.copyto_generic = []
        self.copyto_fname = []

    def create(self):
        """
        Main loop for creating the pbs scripts, based on cases, which
        contains the case name as key and tag dictionairy as value
        """

        # dynamically set walltime based on the number of time steps
        # for thyra, make a list so we base the walltime on the slowest case
        self.nr_time_steps = []
        self.duration = []
        self.t0 = []
        # '[time_stop]' '[dt_sim]'

        # number the pbs jobs:
        count2 = self.pbs_start_number
        # initial cpu count is zero
        count1 = 1
        # scan through all the cases
        i, i_tot = 1, len(self.cases)
        ended = True

        for case in self.cases:

            # get a shorter version for the current cases tag_dict:
            tag_dict = self.cases[case]

            # sanitize paths from a list of predefined tags in tag_dict
            self.sanitize_paths(case, tag_dict)

            # group all values loaded from the tag_dict here, to keep overview
            # the directories to SAVE the results/logs/turb files
            # load all relevant dir settings: the result/logfile/turbulence/zip
            # they are now also available for starting() and ending() parts
            hawc2_exe = tag_dict['[hawc2_exe]']
            self.case = case.replace('.htc', '')
            self.sim_id = tag_dict['[sim_id]']
            self.results_dir = tag_dict['[res_dir]']
            self.eigenfreq_dir = tag_dict['[eigenfreq_dir]']
            self.logs_dir = tag_dict['[log_dir]']
            self.animation_dir = tag_dict['[animation_dir]']
            self.TurbDirName = tag_dict['[turb_dir]']
            self.TurbDb = tag_dict['[turb_db_dir]']
            self.wakeDb = tag_dict['[micro_db_dir]']
            self.meandDb = tag_dict['[meander_db_dir]']
            self.WakeDirName = tag_dict['[micro_dir]']
            self.MeanderDirName = tag_dict['[meander_dir]']
            self.ModelZipFile = tag_dict['[model_zip]']
            self.htc_dir = tag_dict['[htc_dir]']
            self.hydro_dir = tag_dict['[hydro_dir]']
            self.mooring_dir = tag_dict['[mooring_dir]']
            self.model_path = tag_dict['[run_dir]']
            self.turb_base_name = tag_dict['[turb_base_name]']
            self.wake_base_name = tag_dict['[micro_base_name]']
            self.meand_base_name = tag_dict['[meander_base_name]']
            self.walltime = tag_dict['[walltime]']
            self.dyn_walltime = tag_dict['[auto_walltime]']
            self.case_duration = tag_dict['[duration]']

            # create the pbs_out_dir if necesary
            try:
                path = os.path.join(tag_dict['[run_dir]'], tag_dict['[pbs_out_dir]'])
                if not os.path.exists(path):
                    os.makedirs(path)
                self.pbs_out_dir = tag_dict['[pbs_out_dir]']
            except:
                pass

            # create pbs_in subdirectories if necessary
            try:
                path = os.path.join(tag_dict['[run_dir]'], tag_dict['[pbs_in_dir]'])
                if not os.path.exists(path):
                    os.makedirs(path)
                self.pbs_in_dir = tag_dict['[pbs_in_dir]']
            except:
                pass

            try:
                self.copyback_files = tag_dict['[copyback_files]']
                self.copyback_frename = tag_dict['[copyback_frename]']
            except KeyError:
                pass

            try:
                self.copyto_generic = tag_dict['[copyto_generic]']
                self.copyto_files = tag_dict['[copyto_files]']
                if not isinstance(self.copyto_generic, list):
                    raise ValueError('[copyto_generic] should be a list')
                if not isinstance(self.copyto_files, list):
                    raise ValueError('[copyto_files] should be a list')
            except KeyError:
                pass

            # one using just one file so it can be used together with the
            # DLC spreadsheets
            try:
                self.copyback_files = [tag_dict['[copyback_f1]']]
                self.copyback_frename = [tag_dict['[copyback_f1_rename]']]
            except KeyError:
                pass
            try:
                self.copyto_generic = [tag_dict['[copyto_generic_f1]']]
                self.copyto_files = [tag_dict['[copyto_f1]']]
                if not isinstance(self.copyto_generic, list):
                    raise ValueError('[copyto_generic] should be a list')
                if not isinstance(self.copyto_files, list):
                    raise ValueError('[copyto_files] should be a list')
            except KeyError:
                pass

            # related to the dynamically setting the walltime
            duration = float(tag_dict['[time_stop]'])
            dt = float(tag_dict['[dt_sim]'])
            self.nr_time_steps.append(duration/dt)
            self.duration.append(float(tag_dict['[duration]']))
            self.t0.append(float(tag_dict['[t0]']))

            if self.verbose:
                print('htc_dir in pbs.create:')
                print(self.htc_dir)
                print(self.model_path)

            # we only start a new case, if we have something that ended before
            # the very first case has to start with starting
            if ended:
                count1 = 1

#                # when jobs depend on other jobs (constant node loading)
#                if self.que_jobdeps:
#                    jobid = self.pref + str(self.jobid_list[i-1])
#                    jobid_dep = self.pref + str(self.jobid_deps[i-1])
#                else:
#                    jobid = self.pref + str(count2)
#                    jobid_dep = None
                if self.short_job_names:
                    jobid = self.pref + str(count2)
                else:
                    jobid = tag_dict['[case_id]']
                if self.pbs_fname_appendix and self.short_job_names:
                    # define the path for the new pbs script
                    pbs_in_fname = "%s_%s.p" % (tag_dict['[case_id]'], jobid)
                else:
                    pbs_in_fname = "%s.p" % (tag_dict['[case_id]'])
                pbs_path = os.path.join(self.model_path, self.pbs_in_dir,
                                        pbs_in_fname)
                # Start a new pbs script, we only need the tag_dict here
                self.starting(tag_dict, jobid)
                ended = False

            # -----------------------------------------------------------------
            # WRITING THE ACTUAL JOB PARAMETERS

            # browse to the current scratch directory
            self.pbs += "\n\n"
            # mark start of single PBS mode
            self.pbs += "# " + "="*78 + "\n"
            # evaluates to true if LAUNCH_PBS_MODE is NOT set
            self.pbs += "# single PBS mode: one case per PBS job\n"
            self.pbs += "# evaluates to true if LAUNCH_PBS_MODE is NOT set\n"
            self.pbs += "if [ -z ${LAUNCH_PBS_MODE+x} ] ; then\n"
            self.pbs += "  echo\n"
            self.pbs += "  echo 'Execute commands on scratch nodes'\n"
            self.pbs += "  cd \"%s/$USER/$PBS_JOBID\"\n" % self.node_run_root
            self.pbs += "  # create unique dir for each CPU\n"
            self.pbs += "  mkdir '%i'; cd '%i'\n" % (count1, count1)

            # output the current scratch directory
            self.pbs += "  pwd\n"
            # zip file has been copied to the node before (in start_pbs())
            # unzip now in the CPU scratch directory (zip file is one level up)
            self.pbs += "  /usr/bin/unzip ../" + self.ModelZipFile + '\n'
            # create all directories, especially relevant if there are case
            # dependent sub directories that are not present in the ZIP file
            self.pbs += "  mkdir -p '" + self.htc_dir + "'\n"
            self.pbs += "  mkdir -p '" + self.results_dir + "'\n"
            self.pbs += "  mkdir -p '" + self.logs_dir + "'\n"
            if self.TurbDirName is not None or self.TurbDirName.lower()!='none':
                self.pbs += "  mkdir -p '" + self.TurbDirName + "'\n"

            if self.WakeDirName and self.WakeDirName != self.TurbDirName:
                if str(self.WakeDirName).lower() != 'none':
                    self.pbs += "  mkdir -p '" + self.WakeDirName + "'\n"

            if self.MeanderDirName and self.MeanderDirName != self.TurbDirName:
                if str(self.MeanderDirName).lower() != 'none':
                    self.pbs += "  mkdir -p '" + self.MeanderDirName + "'\n"

            if self.hydro_dir:
                self.pbs += "  mkdir -p '" + self.hydro_dir + "'\n"
            # create the eigen analysis dir just in case that is necessary
            if self.eigenfreq_dir:
                self.pbs += "  mkdir -p '%s'\n" % self.eigenfreq_dir

            # and copy the htc file to the node
            # use double quotes here because we need to expand $PBS_O_WORKDIR
            case_source = os.path.join('$PBS_O_WORKDIR', self.htc_dir, case)
            self.pbs += "  cp -R \"%s\" '%s'\n" % (case_source, self.htc_dir)

            # if there is a turbulence file data base dir, copy from there
            if self.TurbDb:
                turb_dir_src = os.path.join('$PBS_O_WORKDIR', self.TurbDb)
            else:
                turb_dir_src = os.path.join('$PBS_O_WORKDIR', self.TurbDirName)

            # the original behaviour makes assumptions on the turbulence box
            # names: turb_base_name_xxx_u.bin, turb_base_name_xxx_v.bin
            if self.turb_base_name is not None:
                turb_src = os.path.join(turb_dir_src, self.turb_base_name)
                # double quotes for expanding variables at run time
                self.pbs += "  cp -R \"%s\"*.bin \"%s\"\n" % (turb_src, self.TurbDirName)
            # more generally, literally define the names of the boxes for u,v,w
            # components
            elif '[turb_fname_u]' in tag_dict:
                turb_u = os.path.join(turb_dir_src, tag_dict['[turb_fname_u]'])
                turb_v = os.path.join(turb_dir_src, tag_dict['[turb_fname_v]'])
                turb_w = os.path.join(turb_dir_src, tag_dict['[turb_fname_w]'])
                self.pbs += "  cp \"%s\" \"%s\"\n" % (turb_u, self.TurbDirName)
                self.pbs += "  cp \"%s\" \"%s\"\n" % (turb_v, self.TurbDirName)
                self.pbs += "  cp \"%s\" \"%s\"\n" % (turb_w, self.TurbDirName)

            # if there is a turbulence file data base dir, copy from there
            if self.wakeDb and self.WakeDirName:
                wake_dir_src = os.path.join('$PBS_O_WORKDIR', self.wakeDb)
            elif self.WakeDirName:
                wake_dir_src = os.path.join('$PBS_O_WORKDIR', self.WakeDirName)
            if self.wake_base_name is not None:
                wake_src = os.path.join(wake_dir_src, self.wake_base_name)
                self.pbs += "  cp -R \"%s\"*.bin \"%s\"\n" % (wake_src, self.WakeDirName)

            # if there is a turbulence file data base dir, copy from there
            if self.meandDb and self.MeanderDirName:
                meand_dir_src = os.path.join('$PBS_O_WORKDIR', self.meandDb)
            elif self.MeanderDirName:
                meand_dir_src = os.path.join('$PBS_O_WORKDIR', self.MeanderDirName)
            if self.meand_base_name is not None:
                meand_src = os.path.join(meand_dir_src, self.meand_base_name)
                self.pbs += "  cp -R \"%s\"*.bin \"%s\"\n" % (meand_src, self.MeanderDirName)

            # copy and rename input files with given versioned name to the
            # required non unique generic version
            for fname, fgen in zip(self.copyto_files, self.copyto_generic):
                fname_full = os.path.join('$PBS_O_WORKDIR', fname)
                self.pbs += "  cp -R \"%s\" \"%s\"\n" % (fname_full, fgen)

            # only apply the wine fix in PBS mode
            self.pbs += self.winefix
            # TODO: activate python env, calculate post-processing
#            self.pbs += 'echo `python -c "import wetb; print(wetb.__version__)"`\n'
            # mark end of single PBS mode
            self.pbs += "# " + "="*78 + "\n\n"

            # end of the file copying in PBS mode
            # mark start of find+xargs mode
            self.pbs += "# " + "-"*78 + "\n"
            self.pbs += "# find+xargs mode: 1 PBS job, multiple cases\n"
            self.pbs += "else\n"
            # when in find+xargs mode, browse to the relevant CPU
            self.pbs += "  # with find+xargs we first browse to CPU folder\n"
            self.pbs += "  cd \"$CPU_NR\"\n"
            self.pbs += "fi\n"
            # mark end of find+xargs mode
            self.pbs += "# " + "-"*78 + "\n\n"

            self.pbs += "echo ''\n"
            # mark start of single PBS mode
            self.pbs += "# " + "="*78 + "\n"
            self.pbs += "# single PBS mode: one case per PBS job\n"
            self.pbs += "# evaluates to true if LAUNCH_PBS_MODE is NOT set\n"
            self.pbs += "if [ -z ${LAUNCH_PBS_MODE+x} ] ; then\n"

            self.pbs += "  echo 'execute HAWC2, fork to background'\n"
            # the hawc2 execution commands via wine, in PBS mode fork and wait
            # METHOD MORE GENERAL
            # case contains the htc file name extension, self.case doesn't
            # we include a leading ./ to help HAWC2
            fname_htc = os.path.join(self.htc_dir, case)
            fname_log = os.path.join(self.logs_dir, self.case)
            ext = '.err.out'
            fname_pbs_out = os.path.join(self.pbs_out_dir, self.case + ext)
            execstr = self.exesingle.format(wine=self.wine, case=case,
                                            fname_htc=fname_htc,
                                            hawc2_exe=hawc2_exe)
            self.pbs += self.prelude
            self.pbs += "  %s &\n" % execstr
            # # OLD METHOD
            # param = (self.wine, hawc2_exe, self.htc_dir+case)
            # self.pbs += "  %s %s ./%s &\n" % param

            # FIXME: running post-processing will only work when 1 HAWC2 job
            # per PBS file, otherwise you have to wait for each job to finish
            # first and then run the post-processing for all those cases
            if self.maxcpu == 1:
                self.pbs += "  wait\n"
                if self.pyenv is not None and self.postpro_node:
                    self.pbs += "  echo 'POST-PROCESSING'\n"
                    self.pbs += "  %s %s\n" % (self.pyenv_cmd, self.pyenv)
                    self.pbs += "  "
                    self.checklogs()
                    self.pbs += "  "
                    self.postprocessing()
                    self.pbs += "  source deactivate\n"
            # mark end of single PBS mode
            self.pbs += "# " + "="*78 + "\n\n"
            # mark start of find+xargs mode
            self.pbs += "# " + "-"*78 + "\n"
            self.pbs += "# find+xargs mode: 1 PBS job, multiple cases\n"
            self.pbs += "else\n"
            # numactl --physcpubind=$CPU_NR

            fname_htc = os.path.join(self.htc_dir, case)
            fname_log = os.path.join(self.logs_dir, self.case)
            ext = '.err.out'
            fname_pbs_out = os.path.join(self.pbs_out_dir, self.case + ext)
            execstr = self.exechunks.format(wine=self.wine, case=case,
                                            fname_htc=fname_htc,
                                            hawc2_exe=hawc2_exe,
                                            pbs_out_dir=self.pbs_out_dir,
                                            logs_dir=self.logs_dir,
                                            fname_log=fname_log,
                                            fname_pbs_out=fname_pbs_out,
                                            winenumactl=self.winenumactl)
            self.pbs += "  echo 'execute HAWC2, do not fork and wait'\n"
            self.pbs += "  %s\n" % execstr

            # param = (self.winenumactl, hawc2_exe, self.htc_dir+case,
            #          self.wine_appendix)
            # self.pbs += '  echo "execute HAWC2, do not fork and wait"\n'
            # self.pbs += "  " + ("%s %s ./%s %s" % param).strip() + "\n"
            if self.pyenv is not None and self.postpro_node_zipchunks:
                self.pbs += "  echo 'POST-PROCESSING'\n"
                self.pbs += "  "
                self.checklogs()
                self.pbs += "  "
                self.postprocessing()
            self.pbs += "fi\n"
            # mark end of find+xargs mode
            self.pbs += "# " + "-"*78 + "\n"

            #self.pbs += "wine get_mac_adresses" + '\n'
            # self.pbs += "cp -R ./*.mac  $PBS_O_WORKDIR/." + '\n'
            # -----------------------------------------------------------------

            # and we end when the cpu's per node are full
            if int(count1/self.maxcpu) == 1:
                # write the end part of the pbs script
                self.ending(pbs_path)
                ended = True
                # print progress:
                replace = ((i/self.maxcpu), (i_tot/self.maxcpu), self.walltime)
                if not self.silent:
                    print('pbs script %3i/%i walltime=%s' % replace)

            count2 += 1
            i += 1
            # the next cpu
            count1 += 1

        # it could be that the last node was not fully loaded. In that case
        # we do not have had a succesfull ending, and we still need to finish
        if not ended:
            # write the end part of the pbs script
            self.ending(pbs_path)
            # progress printing
            replace = ( (i/self.maxcpu), (i_tot/self.maxcpu), self.walltime )
            if not self.silent:
                print('pbs script %3i/%i walltime=%s, partially loaded' % replace)
#            print 'pbs progress, script '+format(i/self.maxcpu,'2.0f')\
#                + '/' + format(i_tot/self.maxcpu, '2.0f') \
#                + ' partially loaded...'

    def starting(self, tag_dict, jobid):
        """
        First part of the pbs script
        """

        # a new clean pbs script!

        case_id = tag_dict['[case_id]']
        std_out = os.path.join(self.pbs_out_dir, case_id + ".out")
        std_err = os.path.join(self.pbs_out_dir, case_id + ".err")

#        if self.que_jobdeps:
#            self.pbs += "#PBS -W depend=afterany:%s\n" % jobid_dep
#        else:
#            self.pbs += "#PBS -a [start_time]" + '\n'

        # in case of gorm, we need to make it work correctly. Now each job
        # has a different scratch dir. If we set maxcpu to 12 they all have
        # the same scratch dir. In that case there should be done something
        # differently

        # specify the number of nodes and cpu's per node required
        if self.maxcpu > 1:
            # Number of nodes and cpus per node (ppn)
            lnodes = int(math.ceil(len(self.cases)/float(self.maxcpu)))
            lnodes = 1
            ppn = self.maxcpu
        else:
            ppn = 1
            lnodes = 1
            # Number of nodes and cpus per node (ppn)

        header_jess = """### Standard Output
#PBS -N {jobid}
#PBS -o '{std_out}'
### Standard Error
#PBS -e '{std_err}'
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -l nodes={lnodes}:ppn={ppn}
### Queue name
{pbs_queue_command}
"""

        header_sophia = """#!/bin/bash
#==============================================================================
# ONLY RELEVANT/USED FOR/BY SLURM ON SOPHIA
#==============================================================================
#SBATCH --job-name={jobid}
#SBATCH --output={std_out}
#SBATCH --error={std_err}
#SBATCH --nodes={lnodes}
#SBATCH --ntasks=4
#SBATCH --time=[walltime]
#==============================================================================
"""

        self.pbs = header_jess.format(jobid=jobid, std_out=std_out,
                                      std_err=std_err, ppn=ppn,
                                      pbs_queue_command=self.pbs_queue_command,
                                      lnodes=lnodes)
        # self.pbs += header_sophia.format(jobid=jobid, std_out=std_out,
        #                               std_err=std_err, ppn=ppn,
        #                               pbs_queue_command=self.pbs_queue_command,
        #                               lnodes=lnodes)

        # mark start of single PBS mode
        self.pbs += "\n" + "# " + "="*78 + "\n"

        # ignore all the file copying when running in xargs mode:
        # when varibale LAUNCH_PBS_MODE is set, file copying is ignored
        # and will have to be done elsewhere
        # we do this so the same launch script can be used either with the node
        # scheduler and the PBS system (for example when re-running cases)
        # evaluates to true if LAUNCH_PBS_MODE is NOT set
        self.pbs += "# single PBS mode: one case per PBS job\n"
        self.pbs += "# evaluates to true if LAUNCH_PBS_MODE is NOT set\n"
        self.pbs += "if [ -z ${LAUNCH_PBS_MODE+x} ] ; then\n"

        self.pbs += "  ### Create scratch directory and copy data to it\n"
        # output the current directory
        self.pbs += "  cd \"$PBS_O_WORKDIR\"\n"
        self.pbs += "  echo 'current working dir (pwd):'\n"
        self.pbs += "  pwd\n"
        # # in very special cases you might want the scratch directory be made
        # if self.node_run_root != '/scratch':
        #     self.pbs += '  mkdir -p %s/$USER\n' % self.node_run_root
        #     self.pbs += '  mkdir -p %s/$USER/$PBS_JOBID\n' % self.node_run_root

        # copy the zip files to the scratch dir on the node
        rpl = (self.ModelZipFile, self.node_run_root)
        self.pbs += "  cp -R '%s' \"%s/$USER/$PBS_JOBID\"\n" % rpl
        self.pbs += "fi\n"
        # mark end of single PBS mode
        self.pbs += "# " + "="*78 + "\n"

    def ending(self, pbs_path):
        """
        Last part of the pbs script, including command to write script to disc
        COPY BACK: from node to
        """
        self.pbs += "\n\n"
        self.pbs += "### Epilogue\n"
        # mark start of single PBS mode
        self.pbs += "# " + "="*78 + "\n"
        # evaluates to true if LAUNCH_PBS_MODE is NOT set
        self.pbs += "# single PBS mode: one case per PBS job\n"
        self.pbs += "# evaluates to true if LAUNCH_PBS_MODE is NOT set\n"
        self.pbs += "if [ -z ${LAUNCH_PBS_MODE+x} ] ; then\n"
        self.pbs += "  ### wait for jobs to finish\n"
        self.pbs += "  wait\n"
        self.pbs += "  echo ''\n"
        self.pbs += "  echo 'Copy back from scratch directory'\n"
        for i in range(1, self.maxcpu+1, 1):

            # navigate to the cpu dir on the node
            # The batch system on Gorm allows more than one job per node.
            # Because of this the scratch directory name includes both the
            # user name and the job ID, that is /scratch/$USER/$PBS_JOBID
            self.copyback_all_files("pbs_mode", i)
            # find+xargs mode only makes sense when maxcpu==1, cpu handling
            # for this mode is handled elsewhere
            if self.maxcpu == 1:
                # mark start of find+xargs mode
                self.pbs += "# " + "-"*78 + "\n"
                self.pbs += "# find+xargs mode: 1 PBS job, multiple cases\n"
                self.pbs += "else\n"
                self.copyback_all_files("find+xargs", None)
        self.pbs += "fi\n"

        # base walltime on the longest simulation in the batch
        nr_time_steps = max(self.nr_time_steps)
        # TODO: take into acccount the difference between time steps with
        # and without output. This penelaty also depends on the number of
        # channels outputted. So from 0 until t0 we have no penalty,
        # from t0 until t0+duration we have the output penalty.

        # always a predifined lead time to account for startup losses
        tmax = int(nr_time_steps*self.secperiter*self.iterperstep + self.tlead)
        if self.dyn_walltime:
            dt_seconds = datetime.datetime.fromtimestamp(tmax)
            self.walltime = dt_seconds.strftime('%H:%M:%S')
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        else:
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        # and reset the nr_time_steps list for the next pbs job file
        self.nr_time_steps = []
        self.t0 = []
        self.duration = []

        # TODO: add logfile checking support directly here. In that way each
        # node will do the logfile checking and statistics calculations right
        # after the simulation. Figure out a way how to merge the data from
        # all the different cases afterwards

        self.pbs += "exit\n"

        if self.verbose:
            print("writing pbs script to path: " + pbs_path)

        if self.pbs.find('rm -r') > -1 or self.pbs.find('rm -f') > -1:
            raise UserWarning('Anything that looks like rm -rf is prohibited.')

        # and write the script to a file:
        write_file(pbs_path, self.pbs, 'w')
        # make the string empty again, for memory
        self.pbs = ''

    def copyback_all_files(self, mode, cpu_nr):
        """Copy back all the files from either scratch to run_dir (PBS mode),
        or from CPU sub-directory back to main directory in find+xargs mode.
        """
        if mode=="find+xargs":
            # we use --remove-source-files otherwise all the results will be
            # here twice: in the main directory and in the cpu-sub directories
            foper = "rsync -a --remove-source-files"
            dst = os.path.join('..', self.sim_id, '')
            # copy back to DB dir, and not the scratch dir root
            dst_db = "$PBS_O_WORKDIR"
            cd2model = "  cd \"%s\"\n" % os.path.join(self.node_run_root, '$USER',
                                                      '$PBS_JOBID', '$CPU_NR', '')
            pbs_mode = False
        else:
            foper = "cp -R"
            dst = "$PBS_O_WORKDIR"
            dst_db = dst
            pbs_mode = True
            cd2model = "  cd \"%s\"\n" % os.path.join(self.node_run_root, '$USER',
                                                      '$PBS_JOBID', '%i' % cpu_nr, '')

        # navigate to the cpu dir on the node
        # The batch system on Gorm/Jess allows more than one job per node.
        # Because of this the scratch directory name includes both the
        # user name and the job ID, that is /scratch/$USER/$PBS_JOBID/CPU_NR
        self.pbs += cd2model

        # create the log, res etc dirs in case they do not exist. Only relevant
        # for pbs_mode, they are created in advance in find+xargs
        if pbs_mode:
            mk = "  mkdir -p"
            self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.results_dir))
            self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.logs_dir))
            if self.animation_dir:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.animation_dir))
            if self.copyback_turb and self.TurbDb:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.TurbDb))
            elif self.copyback_turb:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.TurbDirName))
            if self.copyback_turb and self.wakeDb:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.wakeDb))
            elif self.WakeDirName and self.WakeDirName != self.TurbDirName:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.WakeDirName))
            if self.copyback_turb and self.meandDb:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.meandDb))
            elif self.MeanderDirName and self.MeanderDirName != self.TurbDirName:
                self.pbs += "%s \"%s\"\n" % (mk, os.path.join(dst, self.MeanderDirName))

        # and copy the results and log files frome the scratch to dst
        res_dst = os.path.join(dst, self.results_dir, ".")
        # strict on the first path (single quote), allow expanding on the
        # second path (contains variables)
        self.pbs += "  %s '%s.' \"%s\"\n" % (foper, self.results_dir, res_dst)
        log_dst = os.path.join(dst, self.logs_dir, ".")
        self.pbs += "  %s '%s.' \"%s\"\n" % (foper, self.logs_dir, log_dst)
        # in zipchunks mode by default we also copy the time+std out/err to
        # an additional file that is in pbs_out for consistancy with the
        # pbs_mode approach
        if not pbs_mode:
            pbs_dst = os.path.join(dst, self.pbs_out_dir, ".")
            self.pbs += "  %s '%s.' \"%s\"\n" % (foper, self.pbs_out_dir, pbs_dst)
        if self.animation_dir:
            ani_dst = os.path.join(dst, self.animation_dir, ".")
            self.pbs += "  %s '%s.' \"%s\"\n" % (foper, self.animation_dir, ani_dst)

        if self.eigenfreq_dir:
            # just in case the eig dir has subdirs for the results, only
            # select the base path and cp -r will take care of the rest
            p1 = self.eigenfreq_dir.split('/')[0]
            p2 = os.path.join(dst, p1, ".")
            self.pbs += "  cp -R '%s/.' \"%s\"\n" % (p1, p2)
            # for eigen analysis with floater, modes are in root
            eig_dir_sys = os.path.join(dst, self.eigenfreq_dir, 'system/', '.')
            self.pbs += "  mkdir -p \"%s\"\n" % eig_dir_sys
            self.pbs += "  cp -R mode* \"%s\"\n" % eig_dir_sys
            self.pbs += "  %s mode* \"%s\"\n" % (foper, eig_dir_sys)

        # only copy the turbulence files back if they do not exist
        # for all *.bin files on the node
        cmd = '  for i in `ls *.bin`; do  if [ -e "%s" ]; '
        cmd += 'then echo "$i exists no copyback"; else echo "$i copyback"; '
        cmd += 'cp "$i" "%s"; fi; done\n'
        # copy back turbulence file?
        # browse to the node turb dir
        self.pbs += "\n  echo ''\n"
        self.pbs += "  echo 'COPY BACK TURB IF APPLICABLE'\n"
        if self.copyback_turb and self.TurbDb:
            self.pbs += "  cd '%s'\n" % self.TurbDirName
            tmp = (os.path.join(dst_db, self.TurbDb, '$i'),
                   os.path.join(dst_db, self.TurbDb, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model
        elif self.copyback_turb:
            self.pbs += "  cd '%s'\n" % self.TurbDirName
            tmp = (os.path.join(dst, self.TurbDirName, '$i'),
                   os.path.join(dst, self.TurbDirName, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model

        if self.copyback_turb and self.wakeDb:
            self.pbs += "  cd '%s'\n" % self.WakeDirName
            tmp = (os.path.join(dst_db, self.wakeDb, '$i'),
                   os.path.join(dst_db, self.wakeDb, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model
        elif self.copyback_turb and self.WakeDirName:
            self.pbs += "  cd '%s'\n" % self.WakeDirName
            tmp = (os.path.join(dst, self.WakeDirName, '$i'),
                   os.path.join(dst, self.WakeDirName, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model

        if self.copyback_turb and self.meandDb:
            self.pbs += "  cd '%s'\n" % self.MeanderDirName
            tmp = (os.path.join(dst_db, self.meandDb, '$i'),
                   os.path.join(dst_db, self.meandDb, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model
        elif self.copyback_turb and self.MeanderDirName:
            self.pbs += "  cd '%s'\n" % self.MeanderDirName
            tmp = (os.path.join(dst, self.MeanderDirName, '$i'),
                   os.path.join(dst, self.MeanderDirName, ''))
            self.pbs += cmd % tmp
            # and back to normal model root
            self.pbs += cd2model

        self.pbs += "  echo 'END COPY BACK TURB'\n"
        self.pbs += "  echo ''\n\n"

        # copy back any other kind of files, as specified in copyback_files
        self.pbs += "  echo 'COPYBACK [copyback_files]/[copyback_frename]'\n"
        if len(self.copyback_frename) == 0:
            self.copyback_frename = self.copyback_files
        for fname, fnew in zip(self.copyback_files, self.copyback_frename):
            dst_fnew = os.path.join(dst, fnew)
            self.pbs += "  %s '%s' \"%s\"\n" % (foper, fname, dst_fnew)
        self.pbs += "  echo 'END COPYBACK'\n"
        self.pbs += "  echo ''\n"

        if pbs_mode:
            # check what is left
            self.pbs += "  echo ''\n"
            self.pbs += "  echo 'following files are on "
            self.pbs += "node/cpu %i (find .):'\n" % cpu_nr
            self.pbs += "  find .\n"
            self.pbs += "# " + "="*78 + "\n"
        else:
            self.pbs += "# " + "-"*78 + "\n"

    def checklogs(self):
        """
        """
        self.pbs += 'python -c "from wetb.prepost import statsdel; '
        rpl = (os.path.join(self.logs_dir, self.case+'.log'))
        self.pbs += 'statsdel.logcheck(\'%s\')"\n' % rpl

    def postprocessing(self):
        """Run post-processing just after HAWC2 has ran
        """
        self.pbs += 'python -c "from wetb.prepost import statsdel; '
        fsrc = os.path.join(self.results_dir, self.case)
        mstr = ','.join([str(k) for k in self.m])
        rpl = (fsrc, mstr, str(self.case_duration), '.csv')
        self.pbs += ('statsdel.calc(\'%s\', no_bins=46, m=[%s], '
                     'neq=%s, i0=0, i1=None, ftype=\'%s\')"\n' % rpl)

    def check_results(self, cases):
        """
        Cross-check if all simulations on the list have returned a simulation.
        Combine with ErrorLogs to identify which errors occur where.
        """

        cases_fail = {}

        if not self.silent:
            print('checking if all log and result files are present...', end='')

        # check for each case if we have results and a log file
        for cname, case in cases.items():
            run_dir = case['[run_dir]']
            res_dir = case['[res_dir]']
            log_dir = case['[log_dir]']
            # FIXME: HAWC2 outputs result and logfile always in lower cases
            cname_ = cname.replace('.htc', '').lower()
            f_log = os.path.join(run_dir, log_dir, cname_)
            f_res = os.path.join(run_dir, res_dir, cname_)
            if not os.path.exists(f_log + '.log'):
                cases_fail[cname] = copy.copy(cases[cname])
                continue
            try:
                size_sel = os.stat(f_res + '.sel').st_size
                size_dat = os.stat(f_res + '.dat').st_size
            except OSError:
                size_sel = 0
                size_dat = 0
            try:
                size_hdf = os.stat(f_res + '.hdf5').st_size
            except OSError:
                size_hdf = 0
            if (size_sel < 5 or size_dat < 5) and size_hdf < 5:
                cases_fail[cname] = copy.copy(cases[cname])

        if not self.silent:
            print('done!')

        # length will be zero if there are no failures
        return cases_fail

    def sanitize_paths(self, case, tag_dict):
        """Do some checks on the user defined paths
        """
        dirs = ['[hawc2_exe]', '[sim_id]', '[res_dir]', '[eigenfreq_dir]',
                '[log_dir]', '[animation_dir]', '[turb_dir]', '[micro_dir]',
                '[meander_dir]', '[model_zip]', '[htc_dir]', '[hydro_dir]',
                '[mooring_dir]', '[turb_base_name]', '[case_id]'
                '[micro_base_name]', '[meander_base_name]', '[pbs_out_dir]',
                '[pbs_in_dir]']

        # are allowed to have up to 2 ..
        dirs_db = ['[turb_db_dir]', '[micro_db_dir]', '[meander_db_dir]']

        dirlists = ['[copyto_generic]', '[copyto_files]', '[copyback_f1]',
                    '[copyback_f1_rename]', '[copyto_generic_f1]',
                    '[copyto_f1]', '[copyback_files]', '[copyback_frename]']

        # is the only allowed absolute path
        misc.path_sanitize(tag_dict['[run_dir]'], allowabs=True)

        for pathtag in dirs:
            if pathtag not in tag_dict:
                continue
            path = tag_dict[pathtag]
            # Booleans are allowed
            if isinstance(path, bool) or isinstance(path, type(None)):
                path = str(path)
            misc.path_sanitize(path)

        for pathtag in dirs_db:
            if pathtag not in tag_dict:
                continue
            path = tag_dict[pathtag]
            # Booleans are allowed
            if isinstance(path, bool) or isinstance(path, type(None)):
                path = str(path)
            misc.path_sanitize(path, allowdd=True)

        for pathtag in dirlists:
            if pathtag not in tag_dict:
                continue
            for path in tag_dict[pathtag]:
                # Booleans are allowed
                if isinstance(path, bool) or isinstance(path, type(None)):
                    path = str(path)
                misc.path_sanitize(path)


# TODO: rewrite the error log analysis to something better. Take different
# approach: start from the case and see if the results are present. Than we
# also have the tags_dict available when log-checking a certain case
class ErrorLogs(windIO.LogFile):
    """
    Analyse all HAWC2 log files in any given directory
    ==================================================

    Usage:
    logs = ErrorLogs()
    logs.MsgList    : list with the to be checked messages. Add more if required
    logs.ResultFile : name of the result file (default is ErrorLog.csv)
    logs.PathToLogs : specify the directory where the logsfile reside,
                        the ResultFile will be saved in the same directory.
                        It is also possible to give the path of a specific
                        file, the logfile will not be saved in this case. Save
                        when all required messages are analysed with save()
    logs.check() to analyse all the logfiles and create the ResultFile
    logs.save() to save after single file analysis

    logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], [], ...]
    holding the error messages, empty if no err msg found
    will survive as long as the logs object exists. Keep in
    mind that when processing many messages with many error types (as defined)
    in MsgList might lead to an increase in memory usage.

    logs.MsgListLog2 : dict(key=case, value=[found_error, exit_correct]
        where found_error and exit_correct are booleans. Found error will just
        indicate whether or not any error message has been found

    All files in the speficied folder (PathToLogs) will be evaluated.
    When Any item present in MsgList occurs, the line number of the first
    occurance will be displayed in the ResultFile.
    If more messages are required, add them to the MsgList
    """

    # TODO: move to the HAWC2 plugin for cases

    def __init__(self, silent=False, cases=None, resultfile='ErrorLog.csv'):

        # call init from base class
        super(ErrorLogs, self).__init__()

        self.PathToLogs = ''
        self.ResultFile = resultfile
        self.cases = cases
        self.silent = silent

    # TODO: save this not a csv text string but a df_dict, and save as excel
    # and DataFrame!
    def check(self, appendlog=False, save_iter=False):
        """Check all log files that are to be found in the directory
        ErrorLogs.PathToLogs, or check the specific log file if
        ErrorLogs.PathToLogs points to a specific log file.
        """

        # MsgListLog = []
        FileList = []
        # if a directory, load all files first
        if os.path.isdir(self.PathToLogs):

            for files in os.walk(self.PathToLogs):
                FileList.append(files)
            NrFiles = len(FileList[0][2])
        else:
            # simulate one entry on FileList[0][2], give it the file name
            # and save the directory on in self.PathToLogs
            NrFiles = 1
            FileList.append([ [],[],[os.path.basename(self.PathToLogs)] ])
            self.PathToLogs = os.path.dirname(self.PathToLogs)
            single_file = True
        i=1

        # walk trough the files present in the folder path
        for fname in FileList[0][2]:
            # progress indicator
            if NrFiles > 1:
                if not self.silent:
                    print('progress: ' + str(i) + '/' + str(NrFiles))

            # open the current log file
            f_log = os.path.join(self.PathToLogs, fname)

            if self.cases is not None:
                case = self.cases[fname.replace('.log', '.htc')]
            else:
                case = None
            self.readlog(f_log, case=case, save_iter=save_iter)
            i += 1

#            # if no messages are found for the current file, than say so:
#            if len(MsgList2) == len(self.MsgList):
#                tempLog[-1] = 'NO MESSAGES FOUND'

        # if we have only one file, don't save the log file to disk. It is
        # expected that if we analyse many different single files, this will
        # cause a slower script
        if single_file:
            # now we make it available over the object to save and let it grow
            # over many analysis
            # self.MsgListLog = copy.copy(MsgListLog)
            pass
        else:
            self.save(appendlog=appendlog)

    def save(self, appendlog=False, suffix=None):

        contents = self._header()
        contents = self._msglistlog2csv(contents)

        # write csv file to disk, append to facilitate more logfile analysis
        if isinstance(suffix, str):
            tmp = self.ResultFile.replace('.csv', '_%s.csv' % suffix)
            fname = os.path.join(self.PathToLogs, tmp)
        else:
            fname = os.path.join(self.PathToLogs, str(self.ResultFile))
        if not self.silent:
            print('Error log analysis saved at:')
            print(fname)
        if appendlog:
            mode = 'a'
        else:
            mode = 'w'
        with open(fname, mode) as f:
            f.write(contents)


class ModelData(object):
    """
    Second generation ModelData function. The HawcPy version is crappy, buggy
    and not mutch of use in the optimisation context.
    """
    class st_headers(object):
        """
        Indices to the respective parameters in the HAWC2 st data file
        """
        r     = 0
        m     = 1
        x_cg  = 2
        y_cg  = 3
        ri_x  = 4
        ri_y  = 5
        x_sh  = 6
        y_sh  = 7
        E     = 8
        G     = 9
        Ixx   = 10
        Iyy   = 11
        I_p   = 12
        k_x   = 13
        k_y   = 14
        A     = 15
        pitch = 16
        x_e   = 17
        y_e   = 18

    def __init__(self, verbose=False, silent=False):
        self.verbose = verbose
        self.silent = silent
        # define the column width for printing
        self.col_width = 13
        # formatting and precision
        self.float_hi = 9999.9999
        self.float_lo =  0.01
        self.prec_float = ' 9.05f'
        self.prec_exp =   ' 8.04e'
        self.prec_loss = 0.01

        #0 1  2    3    4    5    6    7   8 9 10   11
        #r m x_cg y_cg ri_x ri_y x_sh y_sh E G I_x  I_y
        #12    13  14  15  16  17  18
        #I_p/K k_x k_y A pitch x_e y_e
        # 19 cols
        self.st_column_header_list = ['r', 'm', 'x_cg', 'y_cg', 'ri_x',
            'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'J', 'k_x',
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        self.st_column_header_list_latex = ['r','m','x_{cg}','y_{cg}','ri_x',
            'ri_y', 'x_{sh}','y_{sh}','E', 'G', 'I_x', 'I_y', 'J', 'k_x',
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        self.st_fpm_cols = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch',
                            'x_e', 'y_e', 'E11', 'E12', 'E13', 'E14', 'E15',
                            'E16', 'E22', 'E23', 'E24', 'E25', 'E26', 'E33',
                            'E34', 'E35', 'E36', 'E44', 'E45', 'E46', 'E55',
                            'E56', 'E66']
        # set column names/indeices as class attributes
        for i, col in enumerate(self.st_fpm_cols):
            setattr(self, col, i)

        # make the column header
        self.column_header_line = 19 * self.col_width * '=' + '\n'
        for k in self.st_column_header_list:
            self.column_header_line += k.rjust(self.col_width)
        self.column_header_line += '\n' + (19 * self.col_width * '=') + '\n'

    def fromline(self, line, separator=' '):
        # TODO: move this to the global function space (dav-general-module)
        """
        split a line, but ignore any blank spaces and return a list with only
        the values, not empty places
        """
        # remove all tabs, new lines, etc? (\t, \r, \n)
        line = line.replace('\t',' ').replace('\n','').replace('\r','')
        # trailing and ending spaces
        line = line.strip()
        line = line.split(separator)
        values = []
        for k in range(len(line)):
            if len(line[k]) > 0: #and k == item_nr:
                values.append(line[k])
                # break

        return values

    def load_st(self, file_path, file_name):
        """
        Now a better format: st_dict has following key/value pairs
            'nset'    : total number of sets in the file (int).
                        This should be autocalculated every time when writing
                        a new file.
            '007-000-0' : set number line in one peace
            '007-001-a' : comments for set-subset nr 07-01 (str)
            '007-001-b' : subset nr and number of data points, should be
                        autocalculate every time you generate a file
            '007-001-d' : data for set-subset nr 07-01 (ndarray(n,19))

        NOW WE ONLY CONSIDER SUBSET COMMENTS, SET COMMENTS, HOW ARE THEY
        TREADED NOW??

        st_dict is for easy remaking the same file. We need a different format
        for easy reading the comments as well. For that we have the st_comments
        """

        # TODO: store this in an HDF5 format! This is perfect for that.

        # read all the lines of the file into memory
        self.st_path, self.st_file = file_path, file_name
        FILE = open(os.path.join(file_path, file_name))
        lines = FILE.readlines()
        FILE.close()

        subset = False
        st_dict = dict()
        st_comments = dict()
        for i, line in enumerate(lines):

            # convert line to list space seperated list
            line_list = self.fromline(line)

            # see if the first character is marking something
            if i == 0:
                # it is possible that the NSET line is not defined
                parts = line.split(' ')
                try:
                    for k in range(10):
                        parts.remove(' ') # throws error when can't find
                except ValueError:
                    pass
                # we don't care what is on the nset line, just capture if
                # there are any comments lines
                set_nr = 0
                subset_nr = 0
                st_dict['000-000-0'] = line

            # marks the start of a set
            if line[0] == '#':
                #sett = True
                # first character is the #, the rest is the number
                set_nr = int(line_list[0][1:])
                st_dict['%03i-000-0' % set_nr] = line
                # and reset subset nr to zero now
                subset_nr = 0
                subset_nr_track = 0
                # and comments only format, back to one string
                st_comments['%03i-000-0' % set_nr] = ' '.join(line_list[1:])

            # marks the start of a subset
            elif line[0] == '$':
                subset_nr_track += 1
                subset = True
                subset_nr = int(line_list[0][1:])
                # and comments only format, back to one string
                setid = '%03i-%03i-b' % (set_nr, subset_nr)
                st_comments[setid] = ' '.join(line_list[2:])

                # check if the number read corresponds to tracking
                if subset_nr is not subset_nr_track:
                    msg = 'subset_nr and subset_nr_track do not match'
                    raise UserWarning(msg)

                nr_points = int(line_list[1])
                st_dict[setid] = line
                # prepare read data points
                sub_set_arr = scipy.zeros((nr_points,19), dtype=np.float64)
                # keep track of where we are on the data array, initialize
                # to 0 for starters
                point = 0

            # in case we are not in subset mode, we only have comments left
            elif not subset:
                # FIXME: how are we dealing with set comments now?
                # subset comments are coming before the actual subset
                # so we account them to one set later than we are now
                #if subset_nr > 0 :
                key = '%03i-%03i-a' % (set_nr, subset_nr+1)
                # in case it is not the first comment line
                if key in st_dict: st_dict[key] += line
                else: st_dict[key]  = line
                ## otherwise we have the set comments
                #else:
                    #key = '%03i-%03i-a' % (set_nr, subset_nr)
                    ## in case it is not the first comment line
                    #if st_dict.has_key(key): st_dict[key] += line
                    #else: st_dict[key]  = line

            # in case we have the data points, make sure there are enough
            # data poinst present, raise an error if it doesn't
            elif len(line_list)==19 and subset:
                # we can store it in the array
                sub_set_arr[point,:] = line_list
                # on the last entry:
                if point == nr_points-1:
                    # save to the dict:
                    st_dict['%03i-%03i-d' % (set_nr, subset_nr)]= sub_set_arr
                    # and indicate we're done subsetting, next we can have
                    # either set or subset comments
                    subset = False
                point += 1

            #else:
                #msg='error in st format: don't know where to put current line'
                #raise UserWarning, msg

        self.st_dict = st_dict
        self.st_comments = st_comments

    def _format_nr(self, number):
        """
        Automatic format the number

        prec_loss : float, default=0.01
            acceptible precision loss expressed in %

        """

        # the formatting of the number
        numabs = abs(number)
        # just a float precision defined in self.prec_float
        if (numabs < self.float_hi and numabs > self.float_lo):
            numfor = format(number, self.prec_float)
        # if it is zero, just simply print as 0.0
        elif number == 0.0:
            numfor = format(number, ' 1.1f')
        # exponentional, precision defined in self.prec_exp
        else:
            numfor = format(number, self.prec_exp)

        try:
            loss = 100.0*abs(1 - (float(numfor)/number))
        except ZeroDivisionError:
            if abs(float(numfor)) > 0.00000001:
                msg = 'precision loss, from %1.10f to %s' \
                            % (number, numfor.strip())
                raise ValueError('precesion loss for new st file')
            else:
                loss = 0
        if loss > self.prec_loss:
            msg = 'precision loss, from %1.10f to %s (%f pc)' \
                        % (number, numfor.strip(), loss)
            raise ValueError(msg)

        return numfor

    def write_st(self, file_path, file_name, print_header=False):
        """
        prec_loss : float, default=0.01
            acceptible precision loss expressed in %
        """
        # TODO: implement all the tests when writing on nset, number of data
        # points, subsetnumber sequence etc

        content = ''

        # sort the key list
        keysort = list(self.st_dict.keys())
        keysort.sort()

        for key in keysort:

            # in case we are just printing what was recorded before
            if not key.endswith('d'):
                content += self.st_dict[key]
            # else we have an array
            else:
                # cycle through data points and print them orderly: control
                # precision depending on the number, keep spacing constant
                # so it is easy to read the textfile
                for m in range(self.st_dict[key].shape[0]):
                    for n in range(self.st_dict[key].shape[1]):
                        # TODO: check what do we lose here?
                        # we are coming from a np.float64, as set in the array
                        # but than it will not work with the format()
                        number = float(self.st_dict[key][m,n])
                        numfor = self._format_nr(number)
                        content += numfor.rjust(self.col_width)
                    content += '\n'

                if print_header:
                    content += self.column_header_line

        # and write file to disk again
        FILE = open(file_path + file_name, 'w')
        FILE.write(content)
        FILE.close()
        if not self.silent:
            print('st file written:', file_path + file_name)

    def write_latex(self, fpath, selection=[]):
        """
        Write a table in Latex format based on the data in the st file.

        selection : list
            [ [setnr, subsetnr, table caption], [setnr, subsetnr, caption],...]
            if not specified, all subsets will be plotted

        """

        cols_p1 = ['r [m]', 'm [kg/m]', 'm(ri{_x})^2 [kgNm^2]',
                   'm(ri{_y})^2 [kgNm^2]', 'EI_x [Nm^2]', 'EI_y [Nm^2]',
                   'EA [N]', 'GJ [\\frac{Nm^2}{rad}]']

        cols_p2 = ['r [m]', 'x_cg [m]', 'y_cg [m]', 'x_sh [m]', 'y_sh [m]',
                'x_e [m]', 'y_e [m]', 'k_x [-]', 'k_y [-]', 'pitch [deg]']

        if len(selection) < 1:
            for key in self.st_dict:
                # but now only take the ones that hold data
                if key[-1] == 'd':
                    selection.append([int(key[:3]), int(key[4:7])])

        for i,j, caption in selection:
            # get the data
            try:
                # set comment should be the name of the body
                set_comment = self.st_comments['%03i-000-0' % (i)]
#                subset_comment = self.st_comments['%03i-%03i-b' % (i,j)]
                st_arr = self.st_dict['%03i-%03i-d' % (i,j)]
            except AttributeError:
                msg = 'ModelData object md is not loaded properly'
                raise AttributeError(msg)

            # build the latex table header
#            textable = u"\\begin{table}[b!]\n"
#            textable += u"\\begin{center}\n"
            textable_p1 = "\\centering\n"
            textable_p1 += "\\begin{tabular}"
            # configure the column properties
            tmp = ['C{2.0 cm}' for k in cols_p1]
            tmp = "|".join(tmp)
            textable_p1 += '{|' + tmp + '|}'
            textable_p1 += '\hline\n'
            # add formula mode for the headers
            tmp = []
            for k in cols_p1:
                k1, k2 = k.split(' ')
                tmp.append('$%s$ $%s$' % (k1,k2) )
#            tmp = [u'$%s$' % k for k in cols_p1]
            textable_p1 += ' & '.join(tmp)
            textable_p1 += '\\\\ \n'
            textable_p1 += '\hline\n'

            textable_p2 = "\\centering\n"
            textable_p2 += "\\begin{tabular}"
            # configure the column properties
            tmp = ['C{1.5 cm}' for k in cols_p2]
            tmp = "|".join(tmp)
            textable_p2 += '{|' + tmp + '|}'
            textable_p2 += '\hline\n'
            # add formula mode for the headers
            tmp = []
            for k in cols_p2:
                k1, k2 = k.split(' ')
                tmp.append('$%s$ $%s$' % (k1,k2) )
#            tmp = [u'$%s$ $%s$' % (k1, k2) for k in cols_p2]
            # hack: spread the last element over two lines
#            tmp[-1] = '$pitch$ $[deg]$'
            textable_p2 += ' & '.join(tmp)
            textable_p2 += '\\\\ \n'
            textable_p2 += '\hline\n'

            for row in range(st_arr.shape[0]):
                r    = st_arr[row, self.st_headers.r]
                m    = st_arr[row,self.st_headers.m]
                x_cg = st_arr[row,self.st_headers.x_cg]
                y_cg = st_arr[row,self.st_headers.y_cg]
                ri_x = st_arr[row,self.st_headers.ri_x]
                ri_y = st_arr[row,self.st_headers.ri_y]
                x_sh = st_arr[row,self.st_headers.x_sh]
                y_sh = st_arr[row,self.st_headers.y_sh]
                E    = st_arr[row,self.st_headers.E]
                G    = st_arr[row,self.st_headers.G]
                Ixx  = st_arr[row,self.st_headers.Ixx]
                Iyy  = st_arr[row,self.st_headers.Iyy]
                I_p  = st_arr[row,self.st_headers.I_p]
                k_x  = st_arr[row,self.st_headers.k_x]
                k_y  = st_arr[row,self.st_headers.k_y]
                A    = st_arr[row,self.st_headers.A]
                pitch = st_arr[row,self.st_headers.pitch]
                x_e   = st_arr[row,self.st_headers.x_e]
                y_e   = st_arr[row,self.st_headers.y_e]
                # WARNING: same order as the labels defined in variable "cols"!
                p1 = [r, m, m*ri_x*ri_x, m*ri_y*ri_y, E*Ixx, E*Iyy, E*A,I_p*G]
                p2 = [r, x_cg, y_cg, x_sh, y_sh, x_e, y_e, k_x, k_y, pitch]

                textable_p1 += " & ".join([self._format_nr(k) for k in p1])
                textable_p1 += '\\\\ \n'

                textable_p2 += " & ".join([self._format_nr(k) for k in p2])
                textable_p2 += '\\\\ \n'

            # default caption
            if caption == '':
                caption = 'HAWC2 cross sectional parameters for body: %s' % set_comment

            textable_p1 += "\hline\n"
            textable_p1 += "\end{tabular}\n"
            textable_p1 += "\caption{%s}\n" % caption
#            textable += u"\end{center}\n"
#            textable += u"\end{table}\n"

            fname = '%s-%s-%03i-%03i_p1' % (self.st_file, set_comment, i, j)
            fname = fname.replace('.', '') + '.tex'
            with open(fpath + fname, 'w') as f:
                f.write(textable_p1)

            textable_p2 += "\hline\n"
            textable_p2 += "\end{tabular}\n"
            textable_p2 += "\caption{%s}\n" % caption
#            textable += u"\end{center}\n"
#            textable += u"\end{table}\n"

            fname = '%s-%s-%03i-%03i_p2' % (self.st_file, set_comment, i, j)
            fname = fname.replace('.', '') + '.tex'
            with open(fpath + fname, 'w') as f:
                f.write(textable_p2)


class WeibullParameters(object):

    def __init__(self):
        self.Vin = 4.
        self.Vr = 12.
        self.Vout = 26.
        self.Vref = 50.
        self.Vstep = 2.
        self.shape_k = 2.


# FIXME: Cases has a memory leek somewhere, this whole thing needs to be
# reconsidered and rely on a DataFrame instead of a dict!
class Cases(object):
    """
    Class for the old htc_dict
    ==========================

    Formerly known as htc_dict: a dictionary with on the key a case identifier
    (case name) and the value is a dictionary holding all the different tags
    and value pairs which define the case

    TODO:

    define a public API so that plugin's can be exposed in a standarized way
    using pre defined variables:

    * pandas DataFrame backend instead of a dictionary

    * generic, so not bound to HAWC2. Goal: manage a lot of simulations
      and their corresponding inputs/outus

    * integration with OpenMDAO?

    * case id (hash)

    * case name (which is typically created with variable_tag_name method)

    * results

    * inputs

    * outputs

    a variable tags that has a dictionary mirror for database alike searching

    launch, post_launch, prepare_(re)launch should be methods of this or
    inheret from Cases

    Create a method to add and remove cases from the pool so you can perform
    some analysis on them. Maybe make a GUI that present a list with current
    cases in the pool and than checkboxes to remove them.

    Remove the HAWC2 specific parts to a HAWC2 plugin. The HAWC2 plugin will
    inheret from Cases. Proposed class name: HAWC2Cases, XFOILCases

    Rename cases to pool? A pool contains several cases, mixing several
    sim_id's?

    create a unique case ID based on the hash value of all the tag+values?
    """

    # TODO: add a method that can reload a certain case_dict, you change
    # some parameters for each case (or some) and than launch again

    #def __init__(self, post_dir, sim_id, resdir=False):
    def __init__(self, *args, **kwargs):
        """
        Either load the cases dictionary if post_dir and sim_id is given,
        otherwise the input is a cases dictionary

        Paramters
        ---------

        cases : dict
            The cases dictionary in case there is only one argument

        post_dir : str
            When using two arguments

        sim_id : str or list
            When using two arguments

        resdir : str, default=False

        loadstats : boolean, default=False

        rem_failed : boolean, default=True

        """

        resdir = kwargs.get('resdir', False)
        self.loadstats = kwargs.get('loadstats', False)
        self.rem_failed = kwargs.get('rem_failed', True)
        self.config = kwargs.get('config', {})
        self.complib = kwargs.get('complib', 'blosc')
        # determine the input argument scenario
        if len(args) == 1:
            if type(args[0]).__name__ == 'dict':
                self.cases = args[0]
                sim_id = False
            else:
                raise ValueError('One argument input should be a cases dict')
        elif len(args) == 2:
            self.post_dir = args[0]
            sim_id = args[1]
        else:
            raise ValueError('Only one or two arguments are allowed.')

        # if sim_id is a list, than merge all sim_id's of that list
        if type(sim_id).__name__ == 'list':
            # stats, dynprop and fail are empty dictionaries if they do not
            # exist
            self.merge_sim_ids(sim_id)
            # and define a new sim_id based on all items from the list
            self.sim_id = '_'.join(sim_id)
        # in case we still need to load the cases dict
        elif type(sim_id).__name__ == 'str':
            self.sim_id = sim_id
            self._get_cases_dict(self.post_dir, sim_id)
            # load the statistics if applicable
            if self.loadstats:
                self.stats_df, self.Leq_df, self.AEP_df = self.load_stats()

        # change the results directory if applicable
        if resdir:
            self.change_results_dir(resdir)

#        # try to load failed cases and remove them
#        try:
#            self.load_failed(sim_id)
#            self.remove_failed()
#        except IOError:
#            pass

        #return self.cases

    def select(self, search_keyval=False, search_key=False):
        """
        Select only a sub set of the cases

        Select either search_keyval or search_key. Using both is not supported
        yet. Run select twice to achieve the same effect. If both are False,
        cases will be emptied!

        Parameters
        ----------

        search_keyval : dictionary, default=False
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier

        search_key : dict, default=False
            The key is the string that should either be inclusive (value TRUE)
            or exclusive (value FALSE) in the case key
        """

        db = misc.DictDB(self.cases)
        if search_keyval:
            db.search(search_keyval)
        elif search_key:
            db.search_key(search_keyval)
        else:
            db.dict_sel = {}
        # and remove all keys that are not in the list
        remove = set(self.cases) - set(db.dict_sel)
        for k in remove:
            self.cases.pop(k)

    def launch(self, runmethod='local', verbose=False, copyback_turb=True,
               silent=False, check_log=True):
        """
        Launch all cases
        """

        launch(self.cases, runmethod=runmethod, verbose=verbose, silent=silent,
               check_log=check_log, copyback_turb=copyback_turb)

    def post_launch(self, save_iter=False, pbs_failed_path=False, suffix=None,
                    path_errorlog=None, silent=False):
        """
        Post Launching Maintenance

        check the logs files and make sure result files are present and
        accounted for.

        Parameters
        ----------

        save_iter : boolean, default=False
            Set to True to save the number of iterations per time step in
            *.iter file (in the same folder as the logfile)

        pbs_failed_path : str, default=False
            If not False, specify the path to which the *.p files of the
            failed cases should be copied to. For example, the dlctemplate
            will set this value to "pbs_in_fail".

        path_errorlog : str, default=None
            Root path of the error logfiles. If set to None (default), the
            value set in the [run_dir] tag is used as the root folder of the
            logfiles.

        suffix : str, default=None
            If not None, the suffix will be appended to file name of the error
            log analysis file as follows: "ErrorLog_suffix.csv".
        """
        # TODO: integrate global post_launch in here
        self.cases_fail = post_launch(self.cases, save_iter=save_iter,
                                      suffix=suffix, path_errorlog=path_errorlog)

        if pbs_failed_path is not False:
            copy_pbs_in_failedcases(self.cases_fail, path=pbs_failed_path,
                                    silent=silent)

        if self.rem_failed:
            self.remove_failed()

    def load_case(self, case):
        try:
            iterations = self.load_iterations(case)
        except IOError:
            iterations = None
        res = self.load_result_file(case)
        return res, iterations

    def load_iterations(self, case):

        fp = os.path.join(case['[run_dir]'], case['[iter_dir]'],
                          case['[case_id]'])
        return np.loadtxt(fp + '.iter')

    # TODO: HAWC2 result file reading should be moved to Simulations
    # and we should also switch to faster HAWC2 reading!
    def load_result_file(self, case, _slice=False):
        """
        Set the correct HAWC2 channels

        Parameters
        ----------

        case : dict
            a case dictionary holding all the tags set for this specific
            HAWC2 simulation

        Returns
        -------

        res : object
            A HawcPy LoadResults instance with attributes such as sig, ch_dict,
            and much much more

        """

        respath = os.path.join(case['[run_dir]'], case['[res_dir]'])
        resfile = case['[case_id]']
        self.res = windIO.LoadResults(respath, resfile.lower())
        if not _slice:
            _slice = np.r_[0:len(self.res.sig)]
        self.time = self.res.sig[_slice,0]
        self.sig = self.res.sig[_slice,:]
        self.case = case

        return self.res

    def load_struct_results(self, case, max_modes=500, nrmodes=1000):
        """
        Load the structural analysis result files
        """
        fpath = os.path.join(case['[run_dir]'], case['[eigenfreq_dir]'])

        # BEAM OUTPUT
        fname = '%s_beam_output.txt' % case['[case_id]']
        beam = None

        # BODY OUTPUT
        fname = '%s_body_output.txt' % case['[case_id]']
        body = None

        # EIGEN BODY
        fname = '%s_eigen_body.txt' % case['[case_id]']
        try:
            eigen_body, rs2 = windIO.ReadEigenBody(fpath, fname, debug=False,
                                              nrmodes=nrmodes)
        except Exception as e:
            eigen_body = None
            print('failed to load eigen_body')
            print(e)

        # EIGEN STRUCT
        fname = '%s_eigen_struct.txt' % case['[case_id]']
        try:
            eigen_struct = windIO.ReadEigenStructure(fpath, fname, debug=False,
                                                     max_modes=max_modes)
        except Exception as e:
            eigen_struct = None
            print('failed to load eigen_struct')
            print(e)

        # STRUCT INERTIA
        fname = '%s_struct_inertia.txt' % case['[case_id]']
        struct_inertia = None

        return beam, body, eigen_body, eigen_struct, struct_inertia

    def load_errorlogs(self):
        """Load error log analysis
        """

        fpath = os.path.join(self.post_dir, self.sim_id + '_ErrorLogs.h5')
        try:
            df_err = pd.read_hdf(fpath, 'table')
        except FileNotFoundError:
            df_err = pd.read_csv(fpath.replace('.h5', '.csv'))

        return df_err

    def change_results_dir(self, forcedir, post_dir=False):
        """
        if the post processing concerns simulations done by thyra/gorm, and
        is downloaded locally, change path to results accordingly

        """
        for case in self.cases:
            self.cases[case]['[run_dir]'] = forcedir
            if post_dir:
                self.cases[case]['[post_dir]'] = post_dir

        #return cases

    def _get_cases_dict(self, post_dir, sim_id):
        """
        Load the pickled dictionary containing all the cases and their
        respective tags.

        Returns
        -------

        cases : Cases object
            cases with failures removed. Failed cases are kept in
            self.cases_fail

        """
        self.cases = load_pickled_file(os.path.join(post_dir, sim_id + '.pkl'))
        self.cases_fail = {}

        if self.rem_failed:
            try:
                self.load_failed(sim_id)
                # ditch all the failed cases out of the htc_dict otherwise
                #  we will have fails when reading the results data files
                self.remove_failed()
            except IOError:
                print("couldn't find pickled failed dictionary")

        return

    def cases2df(self):
        """Convert the cases dict to a DataFrame and check data types"""

        tag_set = []

        # maybe some cases have tags that others don't, create a set with
        # all the tags that occur
        for cname, tags in self.cases.items():
            tag_set.extend(list(tags.keys()))
        # also add cname as a tag
        tag_set.append('cname')
        # only unique tags
        tag_set = set(tag_set)
        # and build the df_dict with all the tags
        df_dict = {tag:[] for tag in tag_set}

        for cname, tags in self.cases.items():
            current_tags = set(tags.keys())
            for tag, value in tags.items():
                df_dict[tag].append(value)
            # and the missing ones
            for tag in (tag_set - current_tags):
                df_dict[tag].append('')

        df_dict2 = misc.df_dict_check_datatypes(df_dict)

        return pd.DataFrame(df_dict2)

    def merge_sim_ids(self, sim_id_list, silent=False):
        """
        Load and merge for a list of sim_id's cases, fail, dynprop and stats
        ====================================================================

        For all sim_id's in the sim_id_list the cases, stats, fail and dynprop
        dictionaries are loaded. If one of them doesn't exists, an empty
        dictionary is returned.

        Currently, there is no warning given when a certain case will be
        overwritten upon merging.

        """

        cases_merged = {}
        cases_fail_merged = {}

        for ii, sim_id in enumerate(sim_id_list):

            # TODO: give a warning if we have double entries or not?
            self.sim_id = sim_id
            self._get_cases_dict(self.post_dir, sim_id)
            cases_fail_merged.update(self.cases_fail)

            # and copy to htc_dict_merged. Note that non unique keys will be
            # overwritten: each case has to have a unique name!
            cases_merged.update(self.cases)

            # merge the statistics if applicable
            # self.stats_dict[channels] = df
            if self.loadstats:
                if ii == 0:
                    self.stats_df, self.Leq_df, self.AEP_df = self.load_stats()
                else:
                    tmp1, tmp2, tmp3 = self.load_stats()
                    self.stats_df = self.stats_df.append(tmp1)
                    if isinstance(self.Leq_df, pd.DataFrame):
                        self.Leq_df = self.Leq_df.append(tmp2)
                    if isinstance(self.AEP_df, pd.DataFrame):
                        self.AEP_df = self.AEP_df.append(tmp3)

        self.cases = cases_merged
        self.cases_fail = cases_fail_merged

    def printall(self, scenario, figpath=''):
        """
        For all the cases, get the average value of a certain channel
        """
        self.figpath = figpath

        # plot for each case the dashboard
        for k in self.cases:

            if scenario == 'blade_deflection':
                self.blade_deflection(self.cases[k], self.figpath)

    def diff(self, refcase_dict, cases):
        """
        See wich tags change over the given cases of the simulation object
        """

        # there is only one case allowed in refcase dict
        if not len(refcase_dict) == 1:
            return ValueError, 'Only one case allowed in refcase dict'

        # take an arbritrary case as baseline for comparison
        refcase = refcase_dict[list(refcase_dict.keys())[0]]
        #reftags = sim_dict[refcase]

        diffdict = dict()
        adddict = dict()
        remdict = dict()
        print()
        print('*'*80)
        print('comparing %i cases' % len(cases))
        print('*'*80)
        print()
        # compare each case with the refcase and see if there are any diffs
        for case in sorted(cases.keys()):
            dd = misc.DictDiff(refcase, cases[case])
            diffdict[case] = dd.changed()
            adddict[case] = dd.added()
            remdict[case] = dd.removed()
            print('')
            print('='*80)
            print(case)
            print('='*80)
            for tag in sorted(diffdict[case]):
                print(tag.rjust(20),':', cases[case][tag])

        return diffdict, adddict, remdict

    def blade_deflection(self, case, **kwargs):
        """
        """

        # read the HAWC2 result file
        self.load_result_file(case)

        # select all the y deflection channels
        db = misc.DictDB(self.res.ch_dict)

        db.search({'sensortype' : 'state pos', 'component' : 'z'})
        # sort the keys and save the mean values to an array/list
        chiz, zvals = [], []
        for key in sorted(db.dict_sel.keys()):
            zvals.append(-self.sig[:,db.dict_sel[key]['chi']].mean())
            chiz.append(db.dict_sel[key]['chi'])

        db.search({'sensortype' : 'state pos', 'component' : 'y'})
        # sort the keys and save the mean values to an array/list
        chiy, yvals = [], []
        for key in sorted(db.dict_sel.keys()):
            yvals.append(self.sig[:,db.dict_sel[key]['chi']].mean())
            chiy.append(db.dict_sel[key]['chi'])

        return np.array(zvals), np.array(yvals)

    def find_failed(self, df_cases=None, df_err=None, save=True,
                    rem_failed=None):
        """Given the log file analysis and the Cases tag list, generate a list
        of failed cases. This is usefull when either some cases have been
        re-run or when the post-processing is done at the same time as the
        simulations (e.g. zipchunks approach). Cases for which the elapsted_time
        column of the error logs is 0 or smaller are also considered as failed

        Parameters
        ----------

        df_cases : df, default=None
            If None the current cases dict object is converted to a dataframe
            using self.cases2df()

        df_err : df, default=None
            If None the current error log is converted to a dataframe using
            self.load_errorlogs()

        save : boolean, default=True
            save the failed cases dict using pickle.

        rem_failed : boolean, default=None
            if None the already set value self.rem_failed is used. Otherwise
            self.rem_failed is overwritten. If True, failed cases are removed
            from self.cases.
        """

        if df_cases is None:
            df_cases = self.cases2df()
        if df_err is None:
            df_err = self.load_errorlogs()
        if rem_failed is None:
            rem_failed = self.rem_failed
        else:
            self.rem_failed = rem_failed

        self.cases_fail = {}

        # convert case_id to log file names
        # logids = pd.DataFrame(columns=[''])
        df_cases['logid'] = df_cases['[log_dir]'] + df_cases['[case_id]'] + '.log'
        # remove those cases for which the logfile has not ended with the last
        # statement "Elapsed time", results in value 0
        df_err = df_err[df_err['elapsted_time'] > 0]
        # we only need to merge with errorlogs using a portion of the data
        # join error logs and df_cases on the logid
        df = pd.merge(df_cases[['logid', '[case_id]']], df_err[['file_name']],
                      left_on='logid', right_on='file_name', how='outer')
        # missing files: those present in df_cases but not in the error log
        # this means file_name is a nan or empty
        # logids_missing2 = set(df_cases['logid']) - set(df_err['file_name'])
        logids_missing = df[df['file_name'].isnull() | (df['file_name']=='')]
        for case_id in logids_missing['[case_id]']:
            cname = case_id + '.htc'
            self.cases_fail[cname] = copy.copy(self.cases[cname])

        if rem_failed:
            self.remove_failed()

        if save:
            save_pickle(os.path.join(self.post_dir, self.sim_id + '_fail.pkl'),
                        self.cases_fail)

    def remove_failed(self, verbose=False):

        # don't do anything if there is nothing defined
        if self.cases_fail == None:
            print('no failed cases to remove')
            return

        nr_cases = len(self.cases)
        # ditch all the failed cases out of the htc_dict
        # otherwise we will have fails when reading the results data files
        for k in self.cases_fail:
            try:
                self.cases_fail[k] = copy.copy(self.cases[k])
                del self.cases[k]
                if verbose:
                    print('removed from htc_dict due to error: ' + k)
            except KeyError:
                if verbose:
                    print('WARNING: failed case does not occur in cases')
                    print('   ', k)
        rpl = (len(self.cases_fail), nr_cases)
        print('removed %i failed cases (out of %i)' % rpl)

    def load_failed(self, sim_id):

        fname = os.path.join(self.post_dir, sim_id + '_fail.pkl')
        FILE = open(fname, 'rb')
        self.cases_fail = pickle.load(FILE)
        FILE.close()

    def load_stats(self, **kwargs):
        """
        Load an existing statistcs file

        Parameters
        ----------

        post_dir : str, default=self.post_dir

        sim_id : str, default=self.sim_id

        fpath : str, default=sim_id

        leq : bool, default=False

        columns : list, default=None

        Returns
        -------

        stats_df : pandas.DataFrame

        Leq_df : pandas.DataFrame

        AEP_df : pandas.DataFrame
        """
        post_dir = kwargs.get('post_dir', self.post_dir)
        sim_id = kwargs.get('sim_id', self.sim_id)
        fpath = os.path.join(post_dir, sim_id)
        Leq_df = kwargs.get('leq', False)
        columns = kwargs.get('columns', None)

        try:
            stats_df = pd.read_hdf(fpath + '_statistics.h5', 'table',
                                   columns=columns)
#            FILE = open(post_dir + sim_id + '_statistics.pkl', 'rb')
#            stats_dict = pickle.load(FILE)
#            FILE.close()
        except IOError:
            stats_df = None
            print('NO STATS FOUND FOR', sim_id)

        try:
            AEP_df = pd.read_hdf(fpath + '_AEP.h5', 'table')
        except IOError:
            AEP_df = None
            print('NO AEP FOUND FOR', sim_id)

        if Leq_df:
            try:
                Leq_df = pd.read_hdf(fpath + '_Leq.h5', 'table')
            except IOError:
                Leq_df = None
                print('NO Leq FOUND FOR', sim_id)

        return stats_df, Leq_df, AEP_df

    def statistics(self, new_sim_id=False, silent=False, ch_sel=None,
                   tags=['[seed]','[windspeed]'], calc_mech_power=False,
                   save=True, m=[3, 4, 6, 8, 10, 12], neq=None, no_bins=46,
                   ch_fatigue={}, update=False, add_sensor=None,
                   chs_resultant=[], i0=0, i1=None, saveinterval=1000,
                   csv=True, suffix=None, A=None, add_sigs={},
                   ch_wind=None, save_new_sigs=False, xlsx=False,
                   bearing_damage_lst=()):
        """
        Calculate statistics and save them in a pandas dataframe. Save also
        every 500 cases the statistics file.

        Parameters
        ----------

        ch_sel : list, default=None
            If defined, only add defined channels to the output data frame.
            The list should contain valid channel names as defined in ch_dict.

        tags : list, default=['[seed]','[windspeed]']
            Select which tag values from cases should be included in the
            dataframes. This will help in selecting and identifying the
            different cases.

        ch_fatigue : list, default=[]
            Valid ch_dict channel names for which the equivalent fatigue load
            needs to be calculated. When set to None, ch_fatigue = ch_sel,
            and hence all channels will have a fatigue analysis.

        add_sigs : dict, default={}
            channel name, expression key/value paires. For example,
            'p1-p1-node-002-forcevec-z*3 + p1-p1-node-002-forcevec-y'

        bearing_damage_lst : iterable, default=()
            Input for wetb.fatigue_tools.bearing_damage: angle and moment
            channels of the bearing of interest.

        chs_resultant

        add_sensor

        calc_mech_power

        saveinterval : int, default=1000
            When processing a large number of cases, the statistics file
            will be saved every saveinterval-ed case

        update : boolean, default=False
            Update an existing DataFrame instead of overwriting one. When
            the number of cases is larger then saveinterval, the statistics
            file will be updated every saveinterval-ed case

        suffix : boolean or str, default=False
            When True, the statistics data file will be appended with a suffix
            that corresponds to the index of the last case added. When a string,
            that suffix will be added to the file name (up to but excluding,
            much like range()). Set to True when a large number of cases is
            being considered in order to avoid excessively large DataFrames.

        csv : boolean, default=False
            In addition to a h5 file, save the statistics also in csv format.

        xlsx : boolean, default=False
            In addition to a h5 file, save the statistics also in MS Excel xlsx
            format.

        Returns
        -------

        dfs : dict
            Dictionary of dataframes, where the key is the channel name of
            the output (that was optionally defined in ch_sel), and the value
            is the dataframe containing the statistical values for all the
            different selected cases.

        """

        def add_df_row(df_dict, **kwargs):
            """
            add a new channel to the df_dict format of ch_df
            """
            for col, value in kwargs.items():
                df_dict[col].append(value)
            for col in (self.res.cols - set(kwargs.keys())):
                df_dict[col].append('')
            return df_dict

        # in case the output changes, remember the original ch_sel
        if ch_sel is not None:
            ch_sel_init = ch_sel.copy()
        else:
            ch_sel_init = None

        if ch_fatigue is None:
            ch_fatigue_init = None
        else:
            ch_fatigue_init = ch_fatigue

        # TODO: should the default tags not be all the tags in the cases dict?
        tag_default = ['[case_id]', '[sim_id]']
        tag_chan = 'channel'
        # merge default with other tags
        for tag in tag_default:
            if tag not in tags:
                tags.append(tag)

        # tags can only be unique, when there the same tag appears twice
        # it will break the DataFrame creation
        if len(tags) is not len(set(tags)):
            raise ValueError('tags can only contain unique entries')

        # get some basic parameters required to calculate statistics
        try:
            case = list(self.cases.keys())[0]
        except IndexError:
            print('no cases to select so no statistics, aborting ...')
            return None

        post_dir = self.cases[case]['[post_dir]']
        if not new_sim_id:
            # select the sim_id from a random case
            sim_id = self.cases[case]['[sim_id]']
        else:
            sim_id = new_sim_id

        if not silent:
            nrcases = len(self.cases)
            print('='*79)
            print('statistics for %s, nr cases: %i' % (sim_id, nrcases))

        df_dict = None
        add_stats = True
        # for finding [] tags
        regex = re.compile('(\\[.*?\\])')

        for ii, (cname, case) in enumerate(self.cases.items()):

            # build the basic df_dict if not defined
            if df_dict is None:
                # the dictionary that will be used to create a pandas dataframe
                df_dict = { tag:[] for tag in tags }
                df_dict[tag_chan] = []
                # add more columns that will help with IDing the channel
                df_dict['channel_name'] = []
                df_dict['channel_units'] = []
                df_dict['channel_nr'] = []
                df_dict['channel_desc'] = []
                add_stats = True

            if not silent:
                pc = '%6.2f' % (float(ii)*100.0/float(nrcases))
                pc += ' %'
                print('stats progress: %4i/%i %s | %s' % (ii, nrcases, pc, cname))

            # make sure the selected tags exist
            if len(tags) != len(set(case) and tags):
                raise KeyError('    not all selected tags exist in cases')

            self.load_result_file(case)
            ch_dict_new = {}
            # this is really messy, now we are also in parallal using the
            # channel DataFrame structure
            ch_df_new = {col:[] for col in self.res.cols}
            ch_df_new['ch_name'] = []
            # calculate the statistics values
#            stats = self.res.calc_stats(self.sig, i0=i0, i1=i1)
            i_new_chans = self.sig.shape[1] # self.Nch
            sig_size = self.res.N  # len(self.sig[i0:i1,0])
            new_sigs = np.ndarray((sig_size, 0))

            for name, expr in add_sigs.items():
                channel_tags = regex.findall(expr)
                # replace all sensor names with expressions
                template = "self.sig[:,self.res.ch_dict['{}']['chi']]"
                for chan in channel_tags:
                    # first remove the [] from the tag
                    # FIXME: fails when the same channel occurs more than once
                    expr = expr.replace(chan, chan[1:-1])
                    expr = expr.replace(chan[1:-1], template.format(chan[1:-1]))

                sig_add = np.ndarray((len(self.sig[:,0]), 1))
                sig_add[:,0] = eval(expr)

                ch_dict_new[name] = {}
                ch_dict_new[name]['chi'] = i_new_chans
                ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                   'ch_name':name})
                i_new_chans += 1
                new_sigs = np.append(new_sigs, sig_add, axis=1)

            if add_sensor is not None:
                chi1 = self.res.ch_dict[add_sensor['ch1_name']]['chi']
                chi2 = self.res.ch_dict[add_sensor['ch2_name']]['chi']
                name = add_sensor['ch_name_add']
                factor = add_sensor['factor']
                operator = add_sensor['operator']

                p1 = self.sig[:,chi1]
                p2 = self.sig[:,chi2]
                sig_add = np.ndarray((len(p1), 1))
                if operator == '*':
                    sig_add[:,0] = p1*p2*factor
                elif operator == '/':
                    sig_add[:,0] = factor*p1/p2
                else:
                    raise ValueError('Operator needs to be either * or /')
#                add_stats = self.res.calc_stats(sig_add)
#                add_stats_i = stats['max'].shape[0]
                # add a new channel description for the mechanical power
                ch_dict_new[name] = {}
                ch_dict_new[name]['chi'] = i_new_chans
                ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                   'ch_name':name})
                i_new_chans += 1
                new_sigs = np.append(new_sigs, sig_add, axis=1)
#                # and append to all the statistics types
#                for key, stats_arr in stats.iteritems():
#                    stats[key] = np.append(stats_arr, add_stats[key])

            # calculate the resultants
            sig_resultants = np.ndarray((sig_size, len(chs_resultant)))
            inc = []
            for j, chs in enumerate(chs_resultant):
                sig_res = np.ndarray((sig_size, len(chs)))
                lab = ''
                no_channel = False
                for i, ch in enumerate(chs):
                    # if the channel does not exist, zet to zero
                    try:
                        chi = self.res.ch_dict[ch]['chi']
                        sig_res[:,i] = self.sig[:,chi]
                        no_channel = False
                    except KeyError:
                        no_channel = True
                    lab += ch.split('-')[-1]
                name = '-'.join(ch.split('-')[:-1] + [lab])
                # when on of the components do no exist, we can not calculate
                # the resultant!
                if no_channel:
                    rpl = (name, cname)
                    print('    missing channel, no resultant for: %s, %s' % rpl)
                    continue
                inc.append(j)
                sig_resultants[:,j] = np.sqrt(sig_res*sig_res).sum(axis=1)
#                resultant = np.sqrt(sig_resultants[:,j].reshape(self.res.N, 1))
#                add_stats = self.res.calc_stats(resultant)
#                add_stats_i = stats['max'].shape[0]
                # add a new channel description for this resultant
                ch_dict_new[name] = {}
                ch_dict_new[name]['chi'] = i_new_chans
                ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                   'ch_name':name})
                i_new_chans += 1
                # and append to all the statistics types
#                for key, stats_arr in stats.iteritems():
#                    stats[key] = np.append(stats_arr, add_stats[key])
            if len(chs_resultant) > 0:
                # but only take the channels that where not missing
                new_sigs = np.append(new_sigs, sig_resultants[:,inc], axis=1)

            # calculate mechanical power first before deriving statistics
            # from it
            if calc_mech_power:
                name = 'stats-shaft-power'
                sig_pmech = np.ndarray((sig_size, 1))
                sig_pmech[:,0] = self.shaft_power()
#                P_mech_stats = self.res.calc_stats(sig_pmech)
#                mech_stats_i = stats['max'].shape[0]
                # add a new channel description for the mechanical power
                ch_dict_new[name] = {}
                ch_dict_new[name]['chi'] = i_new_chans
                ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                   'ch_name':name})
                i_new_chans += 1
                new_sigs = np.append(new_sigs, sig_pmech, axis=1)

                # and C_p_mech
                if A is not None:
                    name = 'stats-cp-mech'
                    if ch_wind is None:
                        chiwind = self.res.ch_dict[self.find_windchan_hub()]['chi']
                    else:
                        chiwind = self.res.ch_dict[ch_wind]['chi']
                    wind = self.res.sig[:,chiwind]
                    cp = np.ndarray((sig_size, 1))
                    cp[:,0] = self.cp(-sig_pmech[:,0], wind, A)
                    # add a new channel description for the mechanical power
                    ch_dict_new[name] = {}
                    ch_dict_new[name]['chi'] = i_new_chans
                    ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                       'ch_name':name})
                    i_new_chans += 1
                    new_sigs = np.append(new_sigs, cp, axis=1)

                    try:
                        try:
                            nn_shaft = self.config['nn_shaft']
                        except:
                            nn_shaft = 4

                        chan_t = 'shaft_nonrotate-shaft-node-%3.3i-forcevec-z'%nn_shaft
                        i = self.res.ch_dict[chan_t]['chi']
                        thrust = self.res.sig[:,i]
                        name = 'stats-ct'
                        ct = np.ndarray((sig_size, 1))
                        ct[:,0] = self.ct(thrust, wind, A)
                        ch_dict_new[name] = {}
                        ch_dict_new[name]['chi'] = i_new_chans
                        ch_df_new = add_df_row(ch_df_new, **{'chi':i_new_chans,
                                                           'ch_name':name})
                        i_new_chans += 1
                        new_sigs = np.append(new_sigs, ct, axis=1)
                    except KeyError:
                        print('    can not calculate CT')

                # and append to all the statistics types
#                for key, stats_arr in stats.iteritems():
#                    stats[key] = np.append(stats_arr, P_mech_stats[key])

            if save_new_sigs and new_sigs.shape[1] > 0:
                chis, keys = [], []
                for key, value in ch_dict_new.items():
                    chis.append(value['chi'])
                    keys.append(key)
                # sort on channel number, so it agrees with the new_sigs array
                isort = np.array(chis).argsort()
                keys = np.array(keys)[isort].tolist()
                df_new_sigs = pd.DataFrame(new_sigs, columns=keys)
                respath = os.path.join(case['[run_dir]'], case['[res_dir]'])
                resfile = case['[case_id]']
                fname = os.path.join(respath, resfile + '_postres.csv')
                print('    saving post-processed res: %s...' % fname, end='')
                df_new_sigs.to_csv(fname, sep='\t')
                print('done!')
                del df_new_sigs

            ch_dict = self.res.ch_dict.copy()
            ch_dict.update(ch_dict_new)

#            ch_df = pd.concat([self.res.ch_df, pd.DataFrame(ch_df_new)])

            # put all the extra channels into the results if we want to also
            # be able to calculate the fatigue loads on them.
            self.sig = np.append(self.sig, new_sigs, axis=1)

            # calculate the statistics values
            stats = self.res.calc_stats(self.sig, i0=i0, i1=i1)

            # calculate any bearing damage
            for name, angle_moment_lst in bearing_damage_lst:
                angle_moment_timeseries_lst = []
                for aa, mm in angle_moment_lst:
                    angle = self.sig[:,self.res.ch_dict[aa]['chi']]
                    moment = self.sig[:,self.res.ch_dict[mm]['chi']]
                    angle_moment_timeseries_lst.append((angle, moment))
                stats[name] = bearing_damage(angle_moment_timeseries_lst)

            # Because each channel is a new row, it doesn't matter how many
            # data channels each case has, and this approach does not brake
            # when different cases have a different number of output channels
            # By default, just take all channels in the result file.
            if ch_sel_init is None:
                ch_sel = list(ch_dict.keys())
#                ch_sel = ch_df.unique_ch_name.tolist()
#                ch_sel = [str(k) for k in ch_sel]
                print('    selecting all channels for statistics')

            # calculate the fatigue properties from selected channels
            fatigue, tags_fatigue = {}, []
            if ch_fatigue_init is None:
                ch_fatigue = ch_sel
                print('    selecting all channels for fatigue')
            else:
                ch_fatigue = ch_fatigue_init

            for ch_id in ch_fatigue:
                chi = ch_dict[ch_id]['chi']
                signal = self.sig[:,chi]
                if neq is None:
                    neq_ = float(case['[duration]'])
                else:
                    neq_ = neq
                eq = self.res.calc_fatigue(signal, no_bins=no_bins, neq=neq_,
                                           m=m)

                # save in the fatigue results
                fatigue[ch_id] = {}
                fatigue[ch_id]['neq'] = neq_
                # when calc_fatigue succeeds, we should have as many items
                # as in m
                if len(eq) == len(m):
                    for eq_, m_ in zip(eq, m):
                        fatigue[ch_id]['m=%2.01f' % m_] = eq_
                # when it fails, we get an empty list back
                else:
                    for m_ in m:
                        fatigue[ch_id]['m=%2.01f' % m_] = np.nan

            # build the fatigue tags
            for m_ in m:
                tag = 'm=%2.01f' % m_
                tags_fatigue.append(tag)
            tags_fatigue.append('neq')

            # -----------------------------------------------------------------
            # define the pandas data frame dict on first run
            # -----------------------------------------------------------------
            # Only build the ch_sel collection once. By definition, the
            # statistics, fatigue and htc tags will not change
            if add_stats:
                # statistical parameters
                for statparam in list(stats.keys()):
                    df_dict[statparam] = []
#                # additional tags
#                for tag in tags:
#                    df_dict[tag] = []
                # fatigue data
                for tag in tags_fatigue:
                    df_dict[tag] = []
                add_stats = False

            for ch_id in ch_sel:

                chi = ch_dict[ch_id]['chi']
                # ch_name is not unique anymore, this doesn't work obviously!
                # use the channel index instead, that is unique
#                chi = ch_df[ch_df.unique_ch_name==ch_id].chi.values[0]

                # sig_stat = [(0=value,1=index),statistic parameter, channel]
                # stat params = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max
                # note that min, mean, std, and range are not relevant for index
                # values. Set to zero there.

                # -------------------------------------------------------------
                # Fill in all the values for the current data entry
                # -------------------------------------------------------------

                # the auxiliry columns
                try:
                    name = self.res.ch_details[chi,0]
                    unit = self.res.ch_details[chi,1]
                    desc = self.res.ch_details[chi,2]
                # the new channels from new_sigs are not in here
                except (IndexError, AttributeError) as e:
                    name = ch_id
                    desc = ''
                    unit = ''
                df_dict['channel_name'].append(name)
                df_dict['channel_units'].append(unit)
                df_dict['channel_desc'].append(desc)
                df_dict['channel_nr'].append(chi)

                # each df line is a channel of case that needs to be id-eed
                df_dict[tag_chan].append(ch_id)

                # for all the statistics keys, save the values for the
                # current channel
                for statparam in list(stats.keys()):
                    df_dict[statparam].append(stats[statparam][chi])
                # and save the tags from the input htc file in order to
                # label each different case properly
                for tag in tags:
                    df_dict[tag].append(case[tag])
                # append any fatigue channels if applicable, otherwise nan
                if ch_id in fatigue:
                    for m_fatigue, eq_ in fatigue[ch_id].items():
                        df_dict[m_fatigue].append(eq_)
                else:
                    for tag in tags_fatigue:
                        # TODO: or should this be NaN?
                        df_dict[tag].append(np.nan)
            # when dealing with a lot of cases, save the stats data at
            # intermediate points to avoid memory issues
            if math.fmod(ii+1, saveinterval) == 0.0:
                df_dict2 = self._df_dict_check_datatypes(df_dict)
                # convert, save/update
                if isinstance(suffix, str):
                    ext = suffix
                elif suffix is True:
                    ext = '_%06i' % (ii+1)
                else:
                    ext = ''
#                dfs = self._df_dict_save(df_dict2, post_dir, sim_id, save=save,
#                                         update=update, csv=csv, suffix=ext)
                # TODO: test this first
                fname = os.path.join(post_dir, sim_id + '_statistics' + ext)
                dfs = misc.dict2df(df_dict2, fname, save=save, update=update,
                                   csv=csv, xlsx=xlsx, check_datatypes=False,
                                   complib=self.complib)

                df_dict2 = None
                df_dict = None
                add_stats = True

        # only save again when there is actual data in df_dict
        if df_dict is not None:
            # make consistent data types
            df_dict2 = self._df_dict_check_datatypes(df_dict)
            # convert, save/update
            if isinstance(suffix, str):
                ext = suffix
            elif suffix is True:
                ext = '_%06i' % ii
            else:
                ext = ''
#            dfs = self._df_dict_save(df_dict2, post_dir, sim_id, save=save,
#                                     update=update, csv=csv, suffix=ext)
            # TODO: test this first
            fname = os.path.join(post_dir, sim_id + '_statistics' + ext)
            dfs = misc.dict2df(df_dict2, fname, save=save, update=update,
                               csv=csv, xlsx=xlsx, check_datatypes=False,
                               complib=self.complib)

        return dfs

    def _add2newsigs(self, ch_dict, name, i_new_chans, new_sigs, addendum):

        ch_dict[name] = {}
        ch_dict[name]['chi'] = i_new_chans
        i_new_chans += 1
        return ch_dict, np.append(new_sigs, addendum, axis=1)

    # TODO: use the version in misc instead.
    def _df_dict_save(self, df_dict2, post_dir, sim_id, save=True,
                      update=False, csv=True, suffix=None):
        """
        Convert the df_dict to df and save/update.

        DEPRICATED, use misc.dict2df instead
        """
        if isinstance(suffix, str):
            fpath = os.path.join(post_dir, sim_id + '_statistics' + suffix)
        else:
            fpath = os.path.join(post_dir, sim_id + '_statistics')

        # in case converting to dataframe fails, fall back
        try:
            dfs = pd.DataFrame(df_dict2)
        except Exception as e:

            FILE = open(fpath + '.pkl', 'wb')
            pickle.dump(df_dict2, FILE, protocol=2)
            FILE.close()
            # check what went wrong
            misc.check_df_dict(df_dict2)
            print('failed to convert to data frame, saved as dict')
            raise(e)

#        # apply categoricals to objects
#        for column_name, column_dtype in dfs.dtypes.iteritems():
#            # applying categoricals mostly makes sense for objects
#            # we ignore all others
#            if column_dtype.name == 'object':
#                dfs[column_name] = dfs[column_name].astype('category')

        # and save/update the statistics database
        if save:
            if update:
                print('updating statistics: %s ...' % (post_dir + sim_id), end='')
                try:
                    dfs.to_hdf('%s.h5' % fpath, 'table', mode='r+', append=True,
                               format='table', complevel=9, complib=self.complib)
                except IOError:
                    print('Can not update, file does not exist. Saving instead'
                          '...', end='')
                    dfs.to_hdf('%s.h5' % fpath, 'table', mode='w',
                               format='table', complevel=9, complib=self.complib)
            else:
                print('saving statistics: %s ...' % (post_dir + sim_id), end='')
                if csv:
                    dfs.to_csv('%s.csv' % fpath)
                dfs.to_hdf('%s.h5' % fpath, 'table', mode='w',
                           format='table', complevel=9, complib=self.complib)

            print('DONE!!\n')

        return dfs

    # TODO: use the version in misc instead.
    def _df_dict_check_datatypes(self, df_dict):
        """
        there might be a mix of strings and numbers now, see if we can have
        the same data type throughout a column
        nasty hack: because of the unicode -> string conversion we might not
        overwrite the same key in the dict.

        DEPRICATED, use misc.df_dict_check_datatypes instead
        """
        # FIXME: this approach will result in twice the memory useage though...
        # we can not pop/delete items from a dict while iterating over it
        df_dict2 = {}
        for colkey, col in df_dict.items():
            # if we have a list, convert to string
            if type(col[0]).__name__ == 'list':
                for ii, item in enumerate(col):
                    col[ii] = '*;*'.join(item)
            # if we already have an array (statistics) or a list of numbers
            # do not try to cast into another data type, because downcasting
            # in that case will not raise any exception
            elif type(col[0]).__name__[:3] in ['flo', 'int', 'nda']:
                df_dict2[str(colkey)] = np.array(col)
                continue
            # in case we have unicodes instead of strings, we need to convert
            # to strings otherwise the saved .h5 file will have pickled elements
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.int32)
            except OverflowError:
                try:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.int64)
                except OverflowError:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
            except ValueError:
                try:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
                except ValueError:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.str)
            except TypeError:
                # in all other cases, make sure we have converted them to
                # strings and NOT unicode
                df_dict2[str(colkey)] = np.array(col, dtype=np.str)
            except Exception as e:
                print('failed to convert column %s to single data type' % colkey)
                raise(e)
        return df_dict2

    def fatigue_lifetime(self, dfs, neq_life, res_dir='res/', fh_lst=None,
                         dlc_folder="dlc%s_iec61400-1ed3/", extra_cols=[],
                         save=False, update=False, csv=False, new_sim_id=False,
                         xlsx=False, years=20.0, silent=False):
        """
        Cacluate the fatigue over a selection of cases and indicate how many
        hours each case contributes to its life time.

        This approach can only work reliably if the common DLC folder
        structure is followed. This also means that a 'dlc_config.xlsx' Excel
        file is required in the HAWC2 root directory (as defined in the
        [run_dir] tag).

        Parameters
        ----------

        dfs : DataFrame
            Statistics Pandas DataFrame. When extra_cols is not defined, it
            should only hold the results of one standard organized DLC (one
            turbine, one inflow case).

        neq_life : float
            Reference number of cycles. Usually, neq is either set to 10e6,
            10e7 or 10e8.

        res_dir : str, default='res/'
            Base directory of the results. Results would be located in
            res/dlc_folder/*.sel. Only relevant when fh_lst is None.

        dlc_folder : str, default="dlc%s_iec61400-1ed3/"
            String with the DLC subfolder names. One string substitution is
            required (%s), and should represent the DLC number (withouth comma
            or point). Not relevant when fh_lst is defined.

        extra_cols : list, default=[]
            The included columns are the material constants, and each row is
            a channel. When multiple DLC cases are included in dfs, the user
            has to define additional columns in order to distinguish between
            the DLC cases.

        fh_lst : list, default=None
            Number of hours for each case over its life time. Format:
            [(filename, hours),...] where, filename is the name of the file
            (can be a full path, but only the base path is considered), hours
            is the number of hours over the life time. When fh_lst is set,
            years, res_dir, dlc_folder and dlc_name are not used.

        years : float, default=20
            Total life time expressed in years, only relevant when fh_lst is
            None.

        Returns
        -------

        df_Leq : DataFrame
            Pandas DataFrame with the life time equivalent load for the given
            neq, all the channels, and a range of material parameters m.
        """
        if not silent:
            print('Calculating life time fatigue load')

        if not isinstance(neq_life, float):
            neq_type = type(neq_life).__name__
            msg = 'neq_life (reference nr of cycles for life time fatigue '
            msg += 'load) should be a float instead of %s' % neq_type
            raise ValueError(msg)

        # get some basic parameters required to calculate statistics
        try:
            case = list(self.cases.keys())[0]
        except IndexError:
            if not silent:
                print('no cases to select so no statistics, aborting ...')
            return None
        post_dir = self.cases[case]['[post_dir]']
        if not new_sim_id:
            # select the sim_id from a random case
            sim_id = self.cases[case]['[sim_id]']
        else:
            sim_id = new_sim_id

        # FIXME: for backward compatibility, the column name of the unique
        # channel name has been changed in the past....
        if 'unique_ch_name' in dfs.columns:
            chan_col_name  = 'unique_ch_name'
        else:
            chan_col_name  = 'channel'

        if fh_lst is None:
            # FIXME: wb has overlap with dlc_config.xlsx, and shape_k doesn't
            # seemed to be used by DLCHighLevel
            wb = WeibullParameters()
            if 'Weibull' in self.config:
                for key in self.config['Weibull']:
                    setattr(wb, key, self.config['Weibull'][key])

            # we assume the run_dir (root) is the same every where
            run_dir = self.cases[case]['[run_dir]']
            fname = os.path.join(run_dir, 'dlc_config.xlsx')
            dlc_cfg = dlc.DLCHighLevel(fname, shape_k=wb.shape_k,
                                       fail_on_resfile_not_found=True)
            # if you need all DLCs, make sure to have %s in the file name
            dlc_cfg.res_folder = os.path.join(run_dir, res_dir, dlc_folder)
            # no need to build list of result files, we already have it form
            # the statistics analysis
            # TODO: could be faster if working with df directly, but how to
            # assure you're res_dir is always ending with path separator?
            # only take the values from 1 channel, not all of them!!
            # FIXME: breaks when not all channels are present for all cases !
            # solution: set channel "Time" as a minimum required channel!
            val = dfs[chan_col_name].values[0]
            sel = dfs[dfs[chan_col_name]==val]
            p1, p2 = sel['[res_dir]'].values, sel['[case_id]'].values
            files = [os.path.join(q1, q2) + '.sel' for q1, q2 in zip(p1, p2)]
            fh_lst = dlc_cfg.file_hour_lst(years=years, files=files)

        # now we have a full path to the result files, but we only need the
        # the case_id to indentify the corresponding entry from the statistics
        # DataFrame (exluciding the .sel extension)
        case_ids = [os.path.basename(k[0].replace('.sel', '')) for k in fh_lst]
        hours = [k[1] for k in fh_lst]

        # safe how many hours each case is active for AEP calculations for
        # debugging and inspection reasons.
        # FIXME: this should be somewhere in its own method or something,
        # and duplication with what is in AEP should be removed
        fname = os.path.join(post_dir, sim_id + '_Leq_hourlist')
        dict_Leq_h = {'case_id':case_ids, 'hours':hours}
        df_Leq_h = misc.dict2df(dict_Leq_h, fname, update=update, csv=csv,
                                save=save, check_datatypes=True, xlsx=xlsx,
                                complib=self.complib)

        # ---------------------------------------------------------------------
        # column definitions
        # ---------------------------------------------------------------------
        # available material constants
        ms, cols = [], []
        for key in dfs:
            if key[:2] == 'm=':
                ms.append(key)
        # when multiple DLC cases are included, add extra cols to identify each
        # DLC group. Make a copy, because extra_cols does not get re-initiated
        # when defined as an optional keyword argument
        extra_cols_ = copy.copy(extra_cols + [chan_col_name])
        cols = copy.copy(ms)
        cols.extend(extra_cols_)
        # ---------------------------------------------------------------------

        # Built the DataFrame, we do not have a unqique channel index
        dict_Leq = {col:[] for col in cols}
        # index on case_id on the original DataFrame so we can select accordingly
        dfs = dfs.set_index('[case_id]')
        # which rows to keep: a
        # select for each channel all the cases
        for grname, gr in dfs.groupby(dfs[chan_col_name]):
            # if one m has any nan's, assume none of them are good and throw
            # away
#            if np.isnan(gr[ms[0]].values).any():
#                sel_rows.pop(grname)
#                continue
            # select the cases in the same order as the corresponding hours
            try:
                sel_sort = gr.loc[case_ids]
            except KeyError:
                if not silent:
                    print('    ignore sensor for Leq:', grname)
            for col in extra_cols_:
                # at this stage we already should have one case, so its
                # identifiers should also be.
                val_unique = sel_sort[col].unique()
                if len(val_unique) > 1:
                    print('found %i sets instead of 1:' % len(val_unique))
                    print(val_unique)
                    raise ValueError('For Leq load, the given DataFrame can '
                                     'only hold one complete DLC set.')
                # values of the identifier columns for each case. We do this
                # in case the original dfs holds multiple DLC cases.
                dict_Leq[col].append(sel_sort[col].unique()[0])

            # R_eq is assumed to be expressed as the 1Hz equivalent load
            # where neq is set to the simulation lenght
#            neq_1hz = sel_sort['neq'].values

            for m in ms:
                # sel_sort[m] holds the equivalent loads for each of the DLC
                # cases: such all the different wind speeds for dlc1.2
                m_ = float(m.split('=')[1])
                # do not multi-ply out neq_1hz from R_eq
                R_eq_mod = np.power(sel_sort[m].values, m_)
                # R_eq_mod will have to be scaled from its simulation length
                # to 1 hour (hour distribution is in hours...). Since the
                # simulation time has not been multiplied out of R_eq_mod yet,
                # we can just multiply with 3600 (instead of doing 3600/neq)
                tmp = (R_eq_mod * np.array(hours) * 3600).sum()
                # the effective Leq for each of the material constants
                dict_Leq[m].append(math.pow(tmp/neq_life, 1.0/m_))
                # the following is twice as slow:
                # [i*j for (i,j) in zip(sel_sort[m].values.tolist(),hours)]

#        collens = misc.check_df_dict(dict_Leq)
        # make consistent data types, and convert to DataFrame
        fname = os.path.join(post_dir, sim_id + '_Leq')
        df_Leq = misc.dict2df(dict_Leq, fname, save=save, update=update,
                              csv=csv, check_datatypes=True, xlsx=xlsx,
                              complib=self.complib)

        # only keep the ones that do not have nan's (only works with index)
        return df_Leq

    def AEP(self, dfs, fh_lst=None, ch_powe='DLL-2-inpvec-2', extra_cols=[],
            res_dir='res/', dlc_folder="dlc%s_iec61400-1ed3/", csv=False,
            new_sim_id=False, save=False, update=False, xlsx=False):

        """
        Calculate the Annual Energy Production (AEP) for DLC1.2 cases.

        Parameters
        ----------

        dfs : DataFrame
            Statistics Pandas DataFrame. When extra_cols is not defined, it
            should only hold the results of one standard organized DLC (one
            turbine, one inflow case).

        fh_lst : list, default=None
            Number of hours for each case over its life time. Format:
            [(filename, hours),...] where, filename is the name of the file
            (can be a full path, but only the base path is considered), hours
            is the number of hours over the life time. When fh_lst is set,
            dlc_folder and dlc_name are not used.

        ch_powe : string, default='DLL-2-inpvec-2'

        extra_cols : list, default=[]
            The included column is just the AEP, and each row is
            a channel. When multiple DLC cases are included in dfs, the user
            has to define additional columns in order to distinguish between
            the DLC cases.

        res_dir : str, default='res/'
            Base directory of the results. Results would be located in
            res/dlc_folder/*.sel

        dlc_folder : str, default="dlc%s_iec61400-1ed3/"
            String with the DLC subfolder names. One string substitution is
            required (%s), and should represent the DLC number (withouth comma
            or point). Not relevant when fh_lst is defined.
        """

        # get some basic parameters required to calculate statistics
        try:
            case = list(self.cases.keys())[0]
        except IndexError:
            print('no cases to select so no statistics, aborting ...')
            return None
        post_dir = self.cases[case]['[post_dir]']
        if not new_sim_id:
            # select the sim_id from a random case
            sim_id = self.cases[case]['[sim_id]']
        else:
            sim_id = new_sim_id

        # FIXME: for backward compatibility, the column name of the unique
        # channel name has been changed in the past....
        if 'unique_ch_name' in dfs.columns:
            chan_col_name  = 'unique_ch_name'
        else:
            chan_col_name  = 'channel'

        if fh_lst is None:
            wb = WeibullParameters()
            if 'Weibull' in self.config:
                for key in self.config['Weibull']:
                    setattr(wb, key, self.config['Weibull'][key])

            # we assume the run_dir (root) is the same every where
            run_dir = self.cases[list(self.cases.keys())[0]]['[run_dir]']
            fname = os.path.join(run_dir, 'dlc_config.xlsx')
            dlc_cfg = dlc.DLCHighLevel(fname, shape_k=wb.shape_k)
            # if you need all DLCs, make sure to have %s in the file name
            dlc_cfg.res_folder = os.path.join(run_dir, res_dir, dlc_folder)
            # TODO: could be faster if working with df directly, but how to
            # assure you're res_dir is always ending with path separator?
            # FIXME: breaks when not all channels are present for all cases !
            # solution: set channel "Time" as a minimum required channel!
            val = dfs[chan_col_name].values[0]
            sel = dfs[dfs[chan_col_name]==val]
            p1, p2 = sel['[res_dir]'].values, sel['[case_id]'].values
            files = [os.path.join(q1, q2) + '.sel' for q1, q2 in zip(p1, p2)]
            fh_lst = dlc_cfg.file_hour_lst(years=1.0, files=files)

        # now we have a full path to the result files, but we only need the
        # the case_id to indentify the corresponding entry from the statistics
        # DataFrame (exluciding the .sel extension)
        def basename(k):
            return os.path.basename(k[0].replace('.sel', ''))
        fh_lst_basename = [(basename(k), k[1]) for k in fh_lst]
        # only take dlc12 for power production
        case_ids = [k[0] for k in fh_lst_basename if k[0][:5]=='dlc12']
        hours = [k[1] for k in fh_lst_basename if k[0][:5]=='dlc12']

        # safe how many hours each case is active for AEP calculations for
        # debugging and inspection reasons.
        # FIXME: this should be somewhere in its own method or something,
        # and duplication with what is in fatigue_lifetime should be removed
        fname = os.path.join(post_dir, sim_id + '_AEP_hourlist')
        dict_AEP_h = {'case_id':case_ids, 'hours':hours}
        df_AEP_h = misc.dict2df(dict_AEP_h, fname, update=update, csv=csv,
                                save=save, check_datatypes=True, xlsx=xlsx,
                                complib=self.complib)

        # check if the power channel actually exists first!
        if ch_powe not in dfs[chan_col_name].unique():
            msg = 'The defined channel for the electrical power does not '
            msg =+ 'exist: %s' % ch_powe
            raise UserWarning(msg)

        # and select only the power channels
        dfs_powe = dfs[dfs[chan_col_name]==ch_powe]

        # by default we have AEP as a column
        cols = ['AEP']
        cols.extend(extra_cols)
        # Built the DataFrame, we do not have a unqique channel index
        dict_AEP = {col:[] for col in cols}
        # index on case_id on the original DataFrame so we can select accordingly
        dfs_powe = dfs_powe.set_index('[case_id]')

        # select the cases in the same order as the corresponding hours
        sel_sort = dfs_powe.loc[case_ids]
        for col in extra_cols:
            # at this stage we already should have one case, so its
            # identifiers should also be.
            val_unique = sel_sort[col].unique()
            if len(val_unique) > 1:
                print('found %i sets instead of 1:' % len(val_unique))
                print(val_unique)
                raise ValueError('For AEP, the given DataFrame can only hold'
                                 'one complete DLC set. Make sure to identify '
                                 'the proper extra_cols to identify the '
                                 'different DLC sets.')
            # values of the identifier columns for each case. We do this
            # in case the original dfs holds multiple DLC cases.
            dict_AEP[col].append(sel_sort[col].unique()[0])

        # and the AEP: take the average, multiply with the duration
#        duration = sel_sort['[duration]'].values
#        power_mean = sel_sort['mean'].values
        AEP = (sel_sort['mean'].values * np.array(hours)).sum()
        dict_AEP['AEP'].append(AEP)

        # make consistent data types, and convert to DataFrame
        fname = os.path.join(post_dir, sim_id + '_AEP')
        df_AEP = misc.dict2df(dict_AEP, fname, update=update, csv=csv,
                              save=save, check_datatypes=True, xlsx=xlsx,
                              complib=self.complib)

        return df_AEP

    def stats2dataframe(self, ch_sel=None, tags=['[seed]','[windspeed]']):
        """
        Convert the archaic statistics dictionary of a group of cases to
        a more convienent pandas dataframe format.

        DEPRICATED, use statistics instead!!

        Parameters
        ----------

        ch_sel : dict, default=None
            Map short names to the channel id's defined in ch_dict in order to
            have more human readable column names in the pandas dataframe. By
            default, if ch_sel is None, a dataframe for each channel in the
            ch_dict (so in the HAWC2 output) will be created. When ch_sel is
            defined, only those channels are considered.
            ch_sel[short name] = full ch_dict identifier

        tags : list, default=['[seed]','[windspeed]']
            Select which tag values from cases should be included in the
            dataframes. This will help in selecting and identifying the
            different cases.

        Returns
        -------

        dfs : dict
            Dictionary of dataframes, where the key is the channel name of
            the output (that was optionally defined in ch_sel), and the value
            is the dataframe containing the statistical values for all the
            different selected cases.
        """

        df_dict = {}

        for cname, case in self.cases.items():

            # make sure the selected tags exist
            if len(tags) != len(set(case) and tags):
                raise KeyError('not all selected tags exist in cases')

            sig_stats = self.stats_dict[cname]['sig_stats']
            ch_dict = self.stats_dict[cname]['ch_dict']

            if ch_sel is None:
                ch_sel = { (i, i) for i in ch_dict }

            for ch_short, ch_name in ch_sel.items():

                chi = ch_dict[ch_name]['chi']
                # sig_stat = [(0=value,1=index),statistic parameter, channel]
                # stat params = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max
                # note that min, mean, std, and range are not relevant for index
                # values. Set to zero there.
                try:
                    df_dict[ch_short]['case name'].append(cname)
                    df_dict[ch_short]['max'].append(   sig_stats[0,0,chi])
                    df_dict[ch_short]['min'].append(   sig_stats[0,1,chi])
                    df_dict[ch_short]['mean'].append(  sig_stats[0,2,chi])
                    df_dict[ch_short]['std'].append(   sig_stats[0,3,chi])
                    df_dict[ch_short]['range'].append( sig_stats[0,4,chi])
                    df_dict[ch_short]['absmax'].append(sig_stats[0,5,chi])
                    for tag in tags:
                        df_dict[ch_short][tag].append(case[tag])
                except KeyError:
                    df_dict[ch_short] = {'case name' : [cname]}
                    df_dict[ch_short]['max']    = [sig_stats[0,0,chi]]
                    df_dict[ch_short]['min']    = [sig_stats[0,1,chi]]
                    df_dict[ch_short]['mean']   = [sig_stats[0,2,chi]]
                    df_dict[ch_short]['std']    = [sig_stats[0,3,chi]]
                    df_dict[ch_short]['range']  = [sig_stats[0,4,chi]]
                    df_dict[ch_short]['absmax'] = [sig_stats[0,5,chi]]
                    for tag in tags:
                        df_dict[ch_short][tag] = [ case[tag] ]

        # and create for each channel a dataframe
        dfs = {}
        for ch_short, df_values in df_dict.items():
            dfs[ch_short] = pd.DataFrame(df_values)

        return dfs

    def load_azimuth(self, azi, load, sectors=360):
        """
        Establish load dependency on rotor azimuth angle
        """

        # sort on azimuth angle
        isort = np.argsort(azi)
        azi = azi[isort]
        load = load[isort]

        azi_sel = np.linspace(0, 360, num=sectors)
        load_sel = np.interp(azi_sel, azi, load)

    def find_windchan_hub(self):
        """
        """
        # if we sort we'll get the largest absolute coordinate last
        for ch in sorted(self.res.ch_dict.keys()):
            if ch[:29] == 'windspeed-global-Vy-0.00-0.00':
                chan_found = ch
        return chan_found

    def ct(self, thrust, wind, A, rho=1.225):
        return thrust / (0.5 * rho * A * wind * wind)

    def cp(self, power, wind, A, rho=1.225):
        return power / (0.5 * rho * A * wind * wind * wind)

    def shaft_power(self):
        """
        Return the mechanical shaft power based on the shaft torsional loading
        """
        try:
            i = self.res.ch_dict['bearing-shaft_rot-angle_speed-rpm']['chi']
            rads = self.res.sig[:,i]*np.pi/30.0
        except KeyError:
            try:
                i = self.res.ch_dict['bearing-shaft_rot-angle_speed-rads']['chi']
                rads = self.res.sig[:,i]
            except KeyError:
                i = self.res.ch_dict['Omega']['chi']
                rads = self.res.sig[:,i]
        try:
            nn_shaft = self.config['nn_shaft']
        except:
            nn_shaft = 4
        itorque = self.res.ch_dict['shaft-shaft-node-%3.3i-momentvec-z'%nn_shaft]['chi']
        torque = self.res.sig[:,itorque]
        # negative means power is being extracted, which is exactly what a wind
        # turbine is about, we call that positive
        return -1.0*torque*rads

    def calc_torque_const(self, save=False, name='ojf'):
        """
        If we have constant RPM over the simulation, calculate the torque
        constant. The current loaded HAWC2 case is considered. Consequently,
        first load a result file with load_result_file

        Parameters
        ----------

        save : boolean, default=False

        name : str, default='ojf'
            File name of the torque constant result. Default to using the
            ojf case name. If set to hawc2, it will the case_id. In both
            cases the file name will be extended with '.kgen'

        Returns
        -------

        [windspeed, rpm, K] : list

        """
        # make sure the results have been loaded previously
        try:
            # get the relevant index to the wanted channels
            # tag: coord-bodyname-pos-sensortype-component
            tag = 'bearing-shaft_nacelle-angle_speed-rpm'
            irpm = self.res.ch_dict[tag]['chi']
            chi_rads = self.res.ch_dict['Omega']['chi']
            tag = 'shaft-shaft-node-001-momentvec-z'
            chi_q = self.res.ch_dict[tag]['chi']
        except AttributeError:
            msg = 'load results first with Cases.load_result_file()'
            raise ValueError(msg)

#        if not self.case['[fix_rpm]']:
#            print
#            return

        windspeed = self.case['[windspeed]']
        rpm = self.res.sig[:,irpm].mean()
        # and get the average rotor torque applied to maintain
        # constant rotor speed
        K = -np.mean(self.res.sig[:,chi_q]*1000./self.res.sig[:,chi_rads])

        result = np.array([windspeed, rpm, K])

        # optionally, save the values and give the case name as file name
        if save:
            fpath = self.case['[post_dir]'] + 'torque_constant/'
            if name == 'hawc2':
                fname = self.case['[case_id]'] + '.kgen'
            elif name == 'ojf':
                fname = self.case['[ojf_case]'] + '.kgen'
            else:
                raise ValueError('name should be either ojf or hawc2')
            # create the torque_constant dir if it doesn't exists
            try:
                os.makedirs(fpath)
            except OSError:
                pass

#            print('gen K saving at:', fpath+fname
            np.savetxt(fpath+fname, result, header='windspeed, rpm, K')

        return result

    def compute_envelopes(self, ch_list, int_env=False, Nx=300):
        """
        The function computes load envelopes for given signals and a single
        load case. Starting from Mx and My moments, the other cross-sectional
        forces are identified.

        Parameters
        ----------

        ch_list : list, list of channels for enevelope computation

        int_env : boolean, default=False
            If the logic parameter is True, the function will interpolate the
            envelope on a given number of points

        Nx : int, default=300
            Number of points for the envelope interpolation

        Returns
        -------

        envelope : dictionary,
            The dictionary has entries refered to the channels selected.
            Inside the dictonary under each entry there is a matrix with 6
            columns, each for the sectional forces and moments

        """

        envelope= {}

        for ch_names in ch_list:
            ichans = []
            for ch_name in ch_names:
                ichans.append(self.res.ch_dict[ch_name]['chi'])
            cloud = self.res.sig[:, ichans]
            # Compute a Convex Hull, the vertices number varies according to
            # the shape of the poligon
            vertices = compute_envelope(cloud, int_env=int_env, Nx=Nx)
            envelope[ch_names[0]] = vertices

        return envelope

    def envelopes(self, silent=False, ch_list=[], append='', int_env=False,
                  Nx=300):
        """
        Calculate envelopes and save them in a table.

        Parameters
        ----------

        silent

        ch_list

        append

        int_env : boolean, default=False
            If the logic parameter is True, the function will interpolate the
            envelope on a given number of points

        Nx : int, default=300
            Number of points for the envelope interpolation

        Returns
        -------


        """
        # get some basic parameters required to calculate statistics
        try:
            case = list(self.cases.keys())[0]
        except IndexError:
            print('no cases to select so no statistics, aborting ...')
            return None

        post_dir = self.cases[case]['[post_dir]']
        sim_id = self.cases[case]['[sim_id]']

        if not silent:
            nrcases = len(self.cases)
            print('='*79)
            print('statistics for %s, nr cases: %i' % (sim_id, nrcases))

        fname = os.path.join(post_dir, sim_id + '_envelope' + append + '.h5')
        h5f = tbl.open_file(fname, mode="w", title=str(sim_id),
                            filters=tbl.Filters(complevel=9))

        # Create a new group under "/" (root)
        for ii, (cname, case) in enumerate(self.cases.items()):

            groupname = str(cname[:-4])
            groupname = groupname.replace('-', '_')
            ctab = h5f.create_group("/", groupname)

            if not silent:
                pc = '%6.2f' % (float(ii)*100.0/float(nrcases))
                pc += ' %'
                print('envelope progress: %4i/%i %s' % (ii, nrcases, pc))

            self.load_result_file(case)

            envelope = self.compute_envelopes(ch_list, int_env=int_env, Nx=Nx)

            for ch_id in ch_list:
                title = str(ch_id[0].replace('-', '_'))
                csv_table = h5f.create_table(ctab, title,
                                             EnvelopeClass.section,
                                             title=title)
                tablerow = csv_table.row
                for row in envelope[ch_id[0]]:
                    tablerow['Mx'] = float(row[0])
                    tablerow['My'] = float(row[1])
                    if len(row)>2:
                        tablerow['Mz'] = float(row[2])
                        if len(row)>3:
                            tablerow['Fx'] = float(row[3])
                            tablerow['Fy'] = float(row[4])
                            tablerow['Fz'] = float(row[5])
                        else:
                            tablerow['Fx'] = 0.0
                            tablerow['Fy'] = 0.0
                            tablerow['Fz'] = 0.0
                    else:
                        tablerow['Mz'] = 0.0
                        tablerow['Fx'] = 0.0
                        tablerow['Fy'] = 0.0
                        tablerow['Fz'] = 0.0
                    tablerow.append()
                csv_table.flush()
        h5f.close()

    def force_lower_case_id(self):
        """Keep for backwards compatibility with the dlctemplate.py
        """
        msg = "force_lower_case_id is depricated and is integrated in "
        msg += "Cases.createcase() instead."
        warnings.warn(msg, DeprecationWarning)

        tmp_cases = {}
        for cname, case in self.cases.items():
             tmp_cases[cname.lower()] = case.copy()
        self.cases = tmp_cases


class EnvelopeClass(object):
    """
    Class with the definition of the table for the envelope results
    """
    class section(tbl.IsDescription):

        Mx = tbl.Float32Col()
        My = tbl.Float32Col()
        Mz = tbl.Float32Col()
        Fx = tbl.Float32Col()
        Fy = tbl.Float32Col()
        Fz = tbl.Float32Col()


class MannTurb64(prepost.PBSScript):
    """
    alfaeps, L, gamma, seed, nr_u, nr_v, nr_w, du, dv, dw high_freq_comp
    mann_turb_x64.exe fname 1.0 29.4 3.0 1209 256 32 32 2.0 5 5 true.

    Following tags have to be defined:
        * [tu_model]
        * [turb_base_name]
        * [MannAlfaEpsilon]
        * [MannL]
        * [MannGamma]
        * [seed]
        * [turb_nr_u]
        * [turb_nr_v]
        * [turb_nr_w]
        * [turb_dx]
        * [turb_dy]
        * [turb_dz]
        * [high_freq_comp]
    """

    def __init__(self, silent=False):
        super(MannTurb64, self).__init__()
        self.exe = 'time WINEARCH=win64 WINEPREFIX=~/.wine wine mann_turb_x64.exe'
        self.winefix = 'winefix\n'
        # PBS configuration
        self.umask = '0003'
        self.walltime = '00:59:59'
        self.queue = 'workq'
        self.lnodes = '1'
        self.ppn = '1'
        self.silent = silent
        self.pbs_in_dir = 'pbs_in_turb/'

    def create_turb(self, base_name, out_base, turb_dir, turb_db_dir, param):
        """

        Parameters
        ----------

        base_name

        out_base

        turb_dir

        turb_db_dir

        param : dictionary
            Should contain the following keys: [MannAlfaEpsilon], [MannL]
            [MannGamma], [seed], [turb_nr_u], [turb_nr_u], [turb_nr_w],
            [turb_dx], [turb_dy], [turb_dz], [high_freq_comp]
        """

        self.path_pbs_e = os.path.join(out_base, turb_dir, base_name + '.err')
        self.path_pbs_o = os.path.join(out_base, turb_dir, base_name + '.out')
        self.path_pbs_i = os.path.join(self.pbs_in_dir, turb_dir, base_name + '.p')

        # apply winefix
        self.prelude = self.winefix
        # browse to scratch dir
        self.prelude += 'cd {}\n'.format(self.scratchdir)

        self.coda = '# COPY BACK FROM SCRATCH AND RENAME, remove _ at end\n'
        # copy back to turb dir at the end
        if turb_db_dir is not None:
            dst = os.path.join('$PBS_O_WORKDIR', turb_db_dir, base_name)
        else:
            dst = os.path.join('$PBS_O_WORKDIR', turb_dir, base_name)
        # FIXME: Mann64 turb exe creator adds an underscore to output
        for comp in list('uvw'):
            src = '{}_{}.bin'.format(base_name, comp)
            dst2 = '{}{}.bin'.format(dst, comp)
            # dst contains $PBS_O_WORKDIR, hence double quote
            self.coda += "cp '{}' \"{}\"\n".format(src, dst2)

        # alfaeps, L, gamma, seed, nr_u, nr_v, nr_w, du, dv, dw high_freq_comp
        rpl = (float(param['[MannAlfaEpsilon]']),
               float(param['[MannL]']),
               float(param['[MannGamma]']),
               int(float(param['[seed]'])),
               int(float(param['[turb_nr_u]'])),
               int(float(param['[turb_nr_v]'])),
               int(float(param['[turb_nr_w]'])),
               float(param['[turb_dx]']),
               float(param['[turb_dy]']),
               float(param['[turb_dz]']),
               int(float(param['[high_freq_comp]'])))
        params = '%1.6f %1.6f %1.6f %i %i %i %i %1.4f %1.4f %1.4f %i' % rpl
        self.execution = "%s '%s' %s" % (self.exe, base_name, params)
        self.create(check_dirs=True)

    def gen_pbs(self, cases):
        """
        Parameters
        ----------

        cases : dict of dicts
            each key holding a dictionary with tag/value pairs.
        """

        case0 = cases[list(cases.keys())[0]]
        # make sure the path's end with a trailing separator, why??
        self.pbsworkdir = os.path.join(case0['[run_dir]'], '')
        if not self.silent:
            print('\nStart creating PBS files for turbulence with Mann64...')

        mann_turb = ['[MannAlfaEpsilon]', '[MannL]', '[MannGamma]',
                     '[turb_nr_u]', '[turb_nr_v]', '[turb_nr_w]', '[seed]',
                     '[turb_dx]', '[turb_dy]', '[turb_dz]',
                     '[high_freq_comp]']
        mann_micro = {k:k.replace(']', '_micro]') for k in mann_turb}
        mann_meander = {k:k.replace(']', '_meander]') for k in mann_turb}

        for cname, case in cases.items():

            # pbs_in/out dir can contain subdirs, only take the base directory
            out_base = misc.path_split_dirs(case['[pbs_out_dir]'])[0]

            # NORMAL ATMOSPHERIC TURBULENCE
            # only relevant for cases with turbulence
            req = '[tu_model]' in case and '[turb_base_name]' in case
            if req and int(case['[tu_model]'])==1:
                base = case['[turb_base_name]']
                # pbs_in/out dir can contain subdirs, only take the base directory
                turb_dir = case['[turb_dir]']
                turb_db_dir = case['[turb_db_dir]']
                # more fail safe switches in case user did not change defaults
                if turb_dir and base and base.lower()!='none' and base!='':
                    self.create_turb(base, out_base, turb_dir, turb_db_dir,
                                     {key:case[key] for key in mann_turb})

            # MICRO TURBULENCE
            if ('[micro_dir]' in case) and ('[micro_base_name]' in case):
                base = case['[micro_base_name]']
                turb_dir = case['[micro_dir]']
                turb_db_dir = case['[micro_db_dir]']
                # more fail safe switches in case user did not change defaults
                if turb_dir and base and base.lower()!='none' and base!='':
                    p = {key:case[mann_micro[key]] for key in mann_turb}
                    self.create_turb(base, out_base, turb_dir, turb_db_dir, p)

            # MEANDER TURBULENCE
            if ('[meander_dir]' in case) and ('[meander_base_name]' in case):
                base = case['[meander_base_name]']
                turb_dir = case['[meander_dir]']
                turb_db_dir = case['[meander_db_dir]']
                # more fail safe switches in case user did not change defaults
                if turb_dir and base and base.lower()!='none' and base!='':
                    p = {key:case[mann_meander[key]] for key in mann_turb}
                    self.create_turb(base, out_base, turb_dir, turb_db_dir, p)


def eigenbody(cases, debug=False):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    This is basically a cases convience wrapper around Hawc2io.ReadEigenBody

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the results, results2 of the eigenvalue
        analysis performed for each body using the following respective
        tags: [eigen_body_results] and [eigen_body_results2].

    """

    #Body data for body number : 3 with the name :nacelle
    #Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = os.path.join(tags['[run_dir]'], tags['[eigenfreq_dir]'])
        # FIXME: do not assuem anything about the file name here, should be
        # fully defined in the tags/dataframe
        file_name = tags['[case_id]'] + '_body_eigen'
        # and load the eigenfrequency body results
        results, results2 = windIO.ReadEigenBody(file_path, file_name,
                                                  nrmodes=10)
        # add them to the htc_dict
        cases[case]['[eigen_body_results]'] = results
        cases[case]['[eigen_body_results2]'] = results2

    return cases


def eigenstructure(cases, debug=False):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    This is basically a cases convience wrapper around
    windIO.ReadEigenStructure

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the modes_arr of the eigenvalue
        analysis performed for the structure.
        The modes array (ndarray(3,n)) holds fd, fn and damping.
    """

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = os.path.join(tags['[run_dir]'], tags['[eigenfreq_dir]'])
        # FIXME: do not assuem anything about the file name here, should be
        # fully defined in the tags/dataframe
        file_name = tags['[case_id]'] + '_strc_eigen'
        # and load the eigenfrequency structure results
        modes = windIO.ReadEigenStructure(file_path, file_name, max_modes=500)
        # add them to the htc_dict
        cases[case]['[eigen_structure]'] = modes

    return cases


if __name__ == '__main__':
    pass
