# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:22:49 2016

@author: dave
"""
# standard python library
import os
from os.path import join as pjoin
import zipfile
import copy
import tarfile
import glob
import shutil
import tempfile

import numpy as np
import pandas as pd
#from tqdm import tqdm

from wetb.prepost.Simulations import Cases
from wetb.prepost import misc


def create_chunks_htc_pbs(cases, sort_by_values=['[Windspeed]'], ppn=20, i0=0,
                          nr_procs_series=9, queue='workq', compress=False,
                          walltime='24:00:00', chunks_dir='zip-chunks-jess',
                          wine_arch='win32', wine_prefix='.wine32',
                          pyenv_cmd='source /home/python/miniconda3/bin/activate',
                          pyenv='py36-wetb', prelude='', ppn_pbs=20):
    """Group a large number of simulations htc and pbs launch scripts into
    different zip files so we can run them with find+xargs on various nodes.
    """

    def chunker(seq, size):
        # for DataFrames you can also use groupby, as taken from:
        # http://stackoverflow.com/a/25703030/3156685
        # for k,g in df.groupby(np.arange(len(df))//10)
        # but this approach is faster, see also:
        # http://stackoverflow.com/a/25701576/3156685
        # http://stackoverflow.com/a/434328/3156685
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def make_zip_chunks(df, ii, sim_id, run_dir, model_zip):
        """Create zip cunks and also create an index
        """

        # create a new zip file, give index of the first element. THis is
        # quasi random due to the sorting we applied earlier
#        ii = df.index[0]
        rpl = (sim_id, ii)
        fname = os.path.join(run_dir, chunks_dir, '%s_chnk_%05i' % rpl)
        zf = zipfile.ZipFile(fname+'.zip', 'w', compression=zipfile.ZIP_STORED)

        # start with appending the base model zip file
        fname_model = os.path.join(run_dir, model_zip)
        with zipfile.ZipFile(fname_model, 'r') as zf_model:
            for n in zf_model.namelist():
                # FIXME: might be duplicates when creating dirs from dirtags
                zf.writestr(n, zf_model.open(n).read())

        # create all necessary directories in the zip file
        dirtags = ['[htc_dir]', '[res_dir]','[log_dir]','[animation_dir]',
                   '[pbs_in_dir]', '[eigenfreq_dir]','[turb_dir]','[micro_dir]',
                   '[meander_dir]','[hydro_dir]', '[mooring_dir]',
                   '[pbs_in_dir]', '[pbs_out_dir]']
        dirnames = []
        for tag in dirtags:
            for dirname in set(df[tag].unique().tolist()):
                if not dirname or dirname.lower() not in ['false', 'none', 0]:
                    dirnames.append(dirname)

        namelist = set(zf.namelist())
        for dirname in set(dirnames):
            if dirname != 0 and pjoin(dirname, '') not in namelist:
                # FIXME: might have duplicates from the base model zip file
                zf.write('.', arcname=os.path.join(dirname, '.'))

        # and the post-processing data
        # FIXME: do not use hard coded paths!
        zf.write('.', arcname='prepost-data/')

        # HTC files
        df_src = df['[run_dir]'] + df['[htc_dir]'] + df['[case_id]']
        df_dst = df['[htc_dir]'] + df['[case_id]']
        # create an index so given the htc file, we can find the chunk nr
        df_index = pd.DataFrame(index=df['[case_id]'].copy(),
                                columns=['chnk_nr', 'name'])
        df_index['chnk_nr'] = ii
        df_index['name'] = os.path.join(chunks_dir, '%s_chnk_%05i' % rpl)
        # Since df_src and df_dst are already Series, iterating is fast an it
        # is slower to first convert to a list
        for src, dst_rel in zip(df_src, df_dst):
            zf.write(src+'.htc', arcname=dst_rel+'.htc')

        # PBS files
        df_src = df['[run_dir]'] + df['[pbs_in_dir]'] + df['[case_id]']
        df_dst = df['[pbs_in_dir]'] + df['[case_id]']
        # Since df_src and df_dst are already Series, iterating is fast an it
        # is slower to first convert to a list
        for src, dst_rel in zip(df_src, df_dst):
            zf.write(src+'.p', arcname=dst_rel+'.p')

        # copy and rename input files with given versioned name to the
        # all files that will have to be renamed to their non-changeable
        # default file name.
        # this is a bit more tricky since unique() will not work on list items
        copyto_files_tmp = df['[copyto_files]'].astype(str)
        copyto_files = []
        # cycle through the unique elements
        for k in set(copyto_files_tmp):
            # k is of form: 'some/file.txt**another/file1.txt
            if len(k) < 1:
                continue
            # note that *;* is done in df_dict_check_datatypes()
            copyto_files.extend(k.split('*;*'))
        # we might still have non unique elements
        copyto_files = set(copyto_files)
        for copyto_file, dst_rel in zip(copyto_files, df_dst):
            src = os.path.join(run_dir, copyto_file)
            # make dir if it does not exist
            namelist = set(zf.namelist())
            # write an empty directory if applicable, make sure ends with /
            copyto_file_folder = pjoin(os.path.dirname(copyto_file), '')
            if len(copyto_file_folder) > 0:
                if copyto_file_folder not in namelist:
                    zf.write('.', arcname=copyto_file_folder)
            # if we have a wildcard, copy all files accordingly
            for fname in glob.glob(copyto_file, recursive=True):
                print(fname)
                if copyto_file not in namelist:
                    zf.write(fname, arcname=fname)
        zf.close()

        return fname, df_index

    pbs_tmplate = "\n"
    pbs_tmplate += "### Standard Output\n"
    pbs_tmplate += "#PBS -N [job_name]\n"
    pbs_tmplate += "#PBS -o [std_out]\n"
    pbs_tmplate += "### Standard Error\n"
    pbs_tmplate += "#PBS -e [std_err]\n"
    pbs_tmplate += "#PBS -W umask=[umask]\n"
    pbs_tmplate += "### Maximum wallclock time format HOURS:MINUTES:SECONDS\n"
    pbs_tmplate += "#PBS -l walltime=[walltime]\n"
    pbs_tmplate += "#PBS -l nodes=[nodes]:ppn=[ppn_pbs]\n"
    pbs_tmplate += "### Queue name\n"
    pbs_tmplate += "#PBS -q [queue]\n"
    pbs_tmplate += "\n"

    # FIXME: this causes troubles on CI runner for the tests (line endings?)
#     pbs_tmplate = """
# ### Standard Output
# #PBS -N [job_name]
# #PBS -o [std_out]
# ### Standard Error
# #PBS -e [std_err]
# #PBS -W umask=[umask]
# ### Maximum wallclock time format HOURS:MINUTES:SECONDS
# #PBS -l walltime=[walltime]
# #PBS -l nodes=[nodes]:ppn=[ppn]
# ### Queue name
# #PBS -q [queue]

# """

    def make_pbs_chunks(df, ii, sim_id, run_dir, model_zip, compress=False,
                        wine_arch='win32', wine_prefix='.wine32'):
        """Create a PBS that:
            * copies all required files (zip chunk) to scratch disk
            * copies all required turbulence files to scratch disk
            * runs everything with find+xargs
            * copies back what's need to mimer
            """
#        ii = df.index[0]
        cmd_find = '/home/MET/sysalt/bin/find'
        cmd_xargs = '/home/MET/sysalt/bin/xargs'
        jobid = '%s_chnk_%05i' % (sim_id, ii)

        # sanitize wine_prefix
        wine_prefix = misc.sanitize_wine_prefix(wine_prefix)

        wineparam = (wine_arch, wine_prefix)

        pbase = os.path.join('/scratch','$USER', '$PBS_JOBID', '')
        post_dir_base = post_dir.split(sim_id)[1]
        if post_dir_base[0] == os.path.sep:
            post_dir_base = post_dir_base[1:]

        # FIXME: commonpath was only added in Python 3.5, but CI runner is old
        try:
            compath = os.path.commonpath
        except AttributeError:
            compath = os.path.commonprefix

        pbs_in_base = compath(df['[pbs_in_dir]'].unique().tolist())
        pbs_in_base = os.path.join(pbs_in_base, '')
        htc_base = compath(df['[htc_dir]'].unique().tolist())
        htc_base = os.path.join(htc_base, '')
        res_base = compath(df['[res_dir]'].unique().tolist())
        res_base = os.path.join(res_base, '')
        log_base = compath(df['[log_dir]'].unique().tolist())
        log_base = os.path.join(log_base, '')

        # =====================================================================
        # PBS HEADER
        pbs = copy.copy(pbs_tmplate)
        pbs = pbs.replace('[job_name]', jobid)
        pbs = pbs.replace('[std_out]', './pbs_out_chunks/%s.out' % jobid)
        pbs = pbs.replace('[std_err]', './pbs_out_chunks/%s.err' % jobid)
        pbs = pbs.replace('[umask]', '0003')
        pbs = pbs.replace('[walltime]', walltime)
        pbs = pbs.replace('[nodes]', str(1)) # only one node for the time being
        pbs = pbs.replace('[ppn_pbs]', str(ppn_pbs))
        pbs = pbs.replace('[queue]', queue)
        pbs += '\necho "%s"\n' % ('-'*70)

        # run prelude code
        # =====================================================================
        pbs += prelude

        # =====================================================================
        # activate the python environment
        if pyenv is not None:
            pbs += 'echo "activate python environment %s"\n' % pyenv
            rpl = (pyenv_cmd, pyenv)
            pbs += '%s %s\n' % rpl
            # sometimes activating an environment fails due to a FileExistsError
            # is this because it is activated at the same time on another node?
            # check twice if the environment got activated for real,
            # but only do so for /home/python/miniconda
            if pyenv_cmd.find('miniconda') > -1:
                pbs += 'echo "CHECK 2x IF %s IS ACTIVE, IF NOT TRY AGAIN"\n' % pyenv
                pbs += 'CMD=\"from distutils.sysconfig import get_python_lib;'
                pbs += 'print (get_python_lib().find(\'/usr/lib/python\'))"\n'
                pbs += 'ACTIVATED=`python -c "$CMD"`\n'
                pbs += 'if [ $ACTIVATED -eq 0 ]; then %s %s;fi\n' % rpl
                pbs += 'ACTIVATED=`python -c "$CMD"`\n'
                pbs += 'if [ $ACTIVATED -eq 0 ]; then %s %s;fi\n' % rpl

        # =====================================================================
        # create all necessary directories at CPU_NR dirs
        # browse to scratch directory
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n\n'
        pbs += 'echo "create CPU directories on the scratch disk"\n'
        pbs += 'mkdir -p "%s"\n' % os.path.join(pbase, sim_id, '')
        for k in range(ppn):
            pbs += 'mkdir -p "%s"\n' % os.path.join(pbase, '%i' % k, '')

        # =====================================================================
        # get the zip-chunk file from the PBS_O_WORKDIR
        pbs += '\n'
        pbs += 'echo "%s"\n' % ('-'*70)
        pbs += 'cd $PBS_O_WORKDIR\n'
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "get the zip-chunk file from the PBS_O_WORKDIR"\n'
        # copy the relevant zip chunk file to the scratch main directory
        rpl = (os.path.join(chunks_dir, jobid), os.path.join(pbase, ''))
        pbs += 'cp "%s.zip" "%s"\n' % rpl

        # =====================================================================
        # unzip to all cpu dirs
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "unzip chunk, create dirs in cpu and sim_id folders"\n'
        # unzip chunk, this contains all relevant folders already, and also
        # contains files defined in [copyto_files]
        for k in list(range(ppn)) + [sim_id]:
            dst = os.path.join('%s' % k, '.')
            pbs += '/usr/bin/unzip "%s" -d "%s" >> /dev/null\n' % (jobid+'.zip', dst)

        # =====================================================================
        # create all turb_db directories
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % os.path.join(pbase, sim_id, '')
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "create turb_db directories"\n'
        turb_db_tags = ['[turb_db_dir]', '[meander_db_dir]', '[micro_db_dir]']
        turb_db_dirs = []
        for tag in turb_db_tags:
            for dirname in set(df[tag].unique().tolist()):
                dirname_s = str(dirname).replace('.', '').replace('/', '')
                if dirname_s.lower() not in ['false', 'none', '0']:
                    turb_db_dirs.append(dirname)
        turb_db_dirs = set(turb_db_dirs)
        # create all turb dirs
        for dirname in sorted(turb_db_dirs):
            pbs += 'mkdir -p "%s"\n' % os.path.join(dirname, '')

        # =====================================================================
        # copy required turbulence from db_dir to scratch/db_dirs
        # turb_db_dir might not be set, same for turb_base_name, for those
        # cases we do not need to copy anything from the database to the node
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "$PBS_O_WORKDIR"\n'
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        base_name_tags = ['[turb_base_name]', '[meander_base_name]',
                          '[micro_base_name]']
        for db, base_name in zip(turb_db_tags, base_name_tags):
            turb_db_dirs = df[db] + df[base_name]
            # When set to None, the DataFrame will have text as None
            # FIXME: CI runner has and old pandas version (v0.14.1)
            try:
                p1 = turb_db_dirs.str.find('None')==-1
                p2 = turb_db_dirs.str.find('none')==-1
                p3 = turb_db_dirs.str.find('false')==-1
                p4 = turb_db_dirs.str.find('False')==-1
#                p4 = turb_db_dirs.str.find('0')==-1
                turb_db_src = turb_db_dirs[p1 & p2 & p3 & p4]
            except AttributeError:
                # and findall returns list with the search str occuring as
                # many times as found in the str...
                # sel should be True if str does NOT occur in turb_db_dirs
                # meaning if findall returns empty list
                sel = [True]*len(turb_db_dirs)
                for val in ['false', 'none', 'None', 'False']:
                    findall = turb_db_dirs.str.findall(val).tolist()
                    # len==0 if nothing has been found
                    sel_ = [True if len(k)==0 else False for k in findall]
                    # merge with other search results, none of the elements
                    # should occur
                    sel = [True if k and kk else False for k, kk in zip(sel, sel_)]
                turb_db_src = turb_db_dirs[sel]
            pbs += '\n'
            pbs += '# copy to scratch db directory for %s, %s\n' % (db, base_name)
            for k in sorted(turb_db_src.unique()):
                dst = os.path.dirname(os.path.join(pbase, sim_id, k))
                # globbing doesn't work in either single- or double-quotes.
                # However, you can interpolate globbing with double-quoted strings
                # https://unix.stackexchange.com/a/67761/163108
                pbs += 'cp "%s"*.bin "%s"\n' % (k, os.path.join(dst, '.'))

        # =====================================================================
        # to be safe, create all turb dirs in the cpu dirs
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % os.path.join(pbase, '')
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "create turb directories in CPU dirs"\n'
        turb_dir_tags = ['[turb_dir]', '[meander_dir]', '[micro_dir]']
        turb_dirs = []
        for tag in turb_dir_tags:
            for dirname in set(df[tag].unique().tolist()):
                dirname_s = str(dirname).replace('.', '').replace('/', '')
                if dirname_s.lower() not in ['false', 'none', '0']:
                    turb_dirs.append(dirname)
        turb_dirs = sorted(set(turb_dirs))
        for k in list(range(ppn)):
            for dirname in turb_dirs:
                pbs += 'mkdir -p "%s"\n' % os.path.join(str(k), dirname, '')

        # =====================================================================
        # symlink everything from the turb_db_dir to the cpu/turb_dir
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % os.path.join(pbase, sim_id, '')
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "Link all turb files into CPU dirs"\n'
        for db_dir, turb_dir in zip(turb_db_tags, turb_dir_tags):
            # FIXME: this needs to be written nicer. We should be able to
            # select from df non-defined values so we can exclude them
            # now it seems they are either None, False or 0 in either
            # boolean, str or int formats
            nogo = ['false', 'none', '0']
            try:
                symlink_dirs = df[db_dir] + '_*_'
                symlink_dirs = symlink_dirs + df[turb_dir]
            except:
                continue
            for symlink in symlink_dirs.unique().tolist():
                db_dir, turb_dir = symlink.split('_*_')
                db_dir_s = db_dir.replace('.', '').replace('/', '').lower()
                turb_dir_s = turb_dir.replace('.', '').replace('/', '').lower()
                if db_dir_s in nogo or turb_dir_s in nogo:
                    continue
                db_dir_abs = os.path.join(pbase, sim_id, db_dir, '')
                for k in list(range(ppn)):
                    turb_dir_abs = os.path.join(pbase, str(k), turb_dir, '')
                    rpl = (db_dir_abs, turb_dir_abs)
                    pbs += 'find "%s" -iname "*.bin" -exec ln -s {} "%s" \\;\n' % rpl

        # copy all from scratch/turb_db to cpu/turb
        # turb_dir_base = os.path.join(compath(list(turb_dirs)), '')
        # pbs += '\necho "%s"\n' % ('-'*70)
        # pbs += 'cd %s\n' % os.path.join(pbase, sim_id, '')
        # pbs += "echo 'current working directory:'\n"
        # pbs += 'pwd\n'
        # pbs += 'echo "Link all turb files into CPU dirs"\n'
        # for k in range(ppn):
        #     rpl = (os.path.relpath(os.path.join(sim_id, turb_dir_base)), k)
        #     pbs += 'find %s -iname "*.bin" -exec cp {} %s/{} \\;\n' % rpl

        # =====================================================================
        # finally we can run find+xargs!!!
        pbs += '\n'
        pbs += 'echo "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "START RUNNING JOBS IN find+xargs MODE"\n'
        if wine_arch!=None or wine_prefix!=None:
            pbs += 'WINEARCH="%s" WINEPREFIX="%s" winefix\n' % wineparam
        pbs += '# run all the PBS *.p files in find+xargs mode\n'
        pbs += 'echo "following cases will be run from following path:"\n'
        pbs += 'echo "%s"\n' % (os.path.join(sim_id, pbs_in_base))
        pbs += 'export LAUNCH_PBS_MODE=false\n'
        rpl = (cmd_find, os.path.join(sim_id, pbs_in_base))
        pbs += "%s '%s' -type f -name '*.p' | sort -z\n" % rpl
        pbs += '\n'
        pbs += 'echo "number of files to be launched: "'
        pbs += '`find "%s" -type f | wc -l`\n' % os.path.join(sim_id, pbs_in_base)
        rpl = (cmd_find, os.path.join(sim_id, pbs_in_base), cmd_xargs, ppn)
        cmd = ("%s '%s' -type f -name '*.p' -print0 | sort -z | %s -0 -I{} "
               "--process-slot-var=CPU_NR -n 1 -P %i sh {}\n" % rpl)
        pbs += cmd
        pbs += 'echo "END OF JOBS IN find+xargs MODE"\n'

        # =====================================================================
        # move results back from the node sim_id dir to the origin
        pbs += '\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'echo "total scratch disk usage:"\n'
        pbs += 'du -hs "%s"\n' % pbase
        pbs += 'cd "%s"\n' % os.path.join(pbase, sim_id)
        pbs += 'echo "current working directory:"\n'
        pbs += 'pwd\n'
        pbs += 'echo "Results saved at sim_id directory:"\n'
        rpl = (os.path.join(pbs_in_base, '*'), os.path.join(htc_base, '*'))
        pbs += 'find .\n'

        if compress:
            # compress all result files into an archive, first *.sel files
            # FIXME: why doesn this work with -name "*.sel" -o -name "*.dat"??
            pbs += '\necho "move results into compressed archive"\n'
            pbs += 'find "%s" -name "*.sel" -print0 ' % res_base
            fname = os.path.join(res_base, 'resfiles_chnk_%05i' % ii)
            pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname
            # now add the *.dat files to the archive
            pbs += 'find "%s" -name "*.dat" -print0 ' % res_base
            fname = os.path.join(res_base, 'resfiles_chnk_%05i' % ii)
            pbs += '| xargs -0 tar --remove-files -rf "%s.tar"\n' % fname
            pbs += 'xz -z2 -T %i "%s.tar"\n' % (ppn, fname)

            # compress all logfiles into an archive
            pbs += '\necho "move logfiles into compressed archive"\n'
            pbs += 'find "%s" -name "*.log" -print0 ' % log_base
            fname = os.path.join(log_base, 'logfiles_chnk_%05i' % ii)
            pbs += '| xargs -0 tar --remove-files -rf "%s.tar"\n' % fname
            pbs += 'xz -z2 -T %i "%s.tar"\n' % (ppn, fname)

        # compress all post-processing results (saved as csv's) into an archive
        pbs += '\necho "move statsdel into compressed archive"\n'
        pbs += 'find "%s" -name "*.csv" -print0 ' % res_base
        fname = os.path.join(post_dir_base, 'statsdel_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf "%s.tar"\n' % fname
        pbs += 'xz -z2 -T %i "%s.tar"\n' % (ppn, fname)

        # compress all post-processing results (saved as csv's) into an archive
        pbs += '\necho "move log analysis into compressed archive"\n'
        pbs += 'find "%s" -name "*.csv" -print0 ' % log_base
        fname = os.path.join(post_dir_base, 'loganalysis_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf "%s.tar"\n' % fname
        pbs += 'xz -z2 -T %i "%s.tar"\n' % (ppn, fname)

        pbs += '\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd "%s"\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "move results back from node scratch/sim_id to origin, '
        pbs += 'but ignore htc, and pbs_in directories."\n'

        tmp = os.path.join(sim_id, '') # make we have trailing /
        pbs += 'echo "copy from %s to $PBS_O_WORKDIR/"\n' % tmp
        pbs += 'time rsync -au "%s" "$PBS_O_WORKDIR/" \\\n' % tmp
        pbs += '    --exclude "%s" \\\n' % os.path.join(pbs_in_base, '*')
        pbs += '    --exclude *.htc\n'
        # when using -u, htc and pbs_in files should be ignored
#        pbs += 'time cp -ru %s $PBS_O_WORKDIR/\n' % tmp
        if pyenv is not None:
            pbs += 'source deactivate\n'
        pbs += 'echo "DONE !!"\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        if wine_arch!=None or wine_prefix!=None:
            pbs += '# in case wine has crashed, kill any remaining wine servers\n'
            pbs += '# caution: ALL the users wineservers will die on this node!\n'
            pbs += 'echo "following wineservers are still running:"\n'
            pbs += 'ps -u $USER -U $USER | grep wineserver\n'
            pbs += 'killall -u $USER wineserver\n'
        pbs += 'exit\n'

        rpl = (sim_id, ii)
        fname = os.path.join(run_dir, chunks_dir, '%s_chnk_%05i' % rpl)

        if pbs.find('rm -rf') > -1 or pbs.find('rm -fr') > -1:
            raise UserWarning('Anything that looks like rm -rf is prohibited.')

        with open(fname+'.p', 'w') as f:
            f.write(pbs)

    def make_pbs_postpro_chunks():
        """When only the post-processing has to be re-done for a chunk.
        """
        pass


    cc = Cases(cases)
    df = cc.cases2df()
    # sort on the specified values in the given columns
    # FIXME: sort_values was only added in Pandas 0.17, but CI runner is old
    try:
        df.sort_values(by=sort_by_values, inplace=True)
    except AttributeError:
        df.sort(columns=sort_by_values, inplace=True)

    # create the directory to store all zipped chunks
    try:
        os.mkdir(os.path.join(df['[run_dir]'].iloc[0], chunks_dir))
    # FIXME: how do you make this work pythonically on both PY2 and PY3?
    except (FileExistsError, OSError):
        pass

    fpath = os.path.join(df['[run_dir]'].iloc[0], 'pbs_out_chunks')
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # remove all cases that start with test_ and move them into a separate
    # chunk
    try:
        sel_notest = df['[Case folder]'].str.find('test_') < 0

        if '[hs2]' in df.columns:
            # depends on if '' or ';' was translated into 0/1 or not
            if df['[hs2]'].dtype == np.dtype('O'):
                sel_hs2 = df['[hs2]'].str.find(';') < 0
            else:
                sel_hs2 = df['[hs2]'] == 1
        else:
            sel_hs2 = pd.Series(data=False, index=df.index)

        # remove hs2 cases from notest
        sel_notest = sel_notest & ~sel_hs2

        # tests but not HS2
        sel_test = ~sel_notest & ~sel_hs2

    except AttributeError:
        # FIXME: CI runner has and old pandas version (v0.14.1)
        # and findall returns list with the search str occuring as
        # many times as found in the str...
        findall = df['[Case folder]'].str.findall('test_').tolist()
        # len==0 if nothing has been found
        sel_notest = [True if len(k)==0 else False for k in findall]

        if '[hs2]' in df.columns:
            if df['[hs2]'].dtype == np.dtype('O'):
                findall = df['[hs2]'].str.findall(';').tolist()
            else:
                findall = (df['[hs2]']==1).tolist()

            # only select it if we do NOT find ;
            sel_hs2 = [True if len(k)==0 else False for k in findall]
        else:
            sel_hs2 = [False]*len(findall)

        # remove hs2 cases from notest
        sel_notest = [i and not j for (i,j) in zip(sel_notest, sel_hs2)]
        # tests but not HS2, so everything that is not in either notest or hs2
        sel_test = [not i and not j for (i,j) in zip(sel_notest, sel_hs2)]

    df_dlc = df[sel_notest]
    df_test = df[sel_test]
    df_hs2 = df[sel_hs2]

    # DLC CHUNKS
    sim_id = df['[sim_id]'].iloc[0]
    run_dir = df['[run_dir]'].iloc[0]
    model_zip = df['[model_zip]'].iloc[0]
    post_dir = df['[post_dir]'].iloc[0]

    names = ['case_id-chunk-index', 'case_id-chunk-test-index',
             'case_id-chunk-hs2-index']
    i0s = [0, 90000, 80000]

    # group test_ hs2 and normal dlc's in 3 different chunks
    for df, name, i02 in zip([df_dlc, df_test, df_hs2], names, i0s):
        i02 = i02 + i0
        df_ind = pd.DataFrame(columns=['chnk_nr'], dtype=np.int32)
        df_ind.index.name = '[case_id]'
        df_iter = chunker(df, nr_procs_series*ppn)
        for ii, dfi in enumerate(df_iter):
            fname, ind = make_zip_chunks(dfi, i02+ii, sim_id, run_dir, model_zip)
            make_pbs_chunks(dfi, i02+ii, sim_id, run_dir, model_zip,
                            wine_arch=wine_arch, wine_prefix=wine_prefix,
                            compress=compress)
            df_ind = df_ind.append(ind)
            print(fname)
        fname = os.path.join(post_dir, 'case_id-chunk-index')
        df_ind['chnk_nr'] = df_ind['chnk_nr'].astype(np.int32)
        df_ind.to_hdf(fname+'.h5', 'table', complevel=9, complib='zlib')
        df_ind.to_csv(fname+'.csv')


def regroup_tarfiles(cc):
    """Re-group all chunks again per [Case folder] compressed file. First all
    chunks are copied to the node scratch disc, then start working on them.
    This only works on a node with PBS stuff.

    Make sure to maintain the same location as defined by the tags!

    [res_dir] and [Case folder] could be multiple directories deep, bu the
    final archive will only contain the files (no directory structure), and
    the name of the archive is that of the last directory:
        /[res_dir]/[Case folder]/[Case folder].tar.xz
        /res/dir/case/folder/dlcname/dlcname.tar.xz

    Parameters
    ----------

    path_pattern : str
        /path/to/files/*.tar.xz

    """

    USER = os.getenv('USER')
    PBS_JOBID = os.getenv('PBS_JOBID')
    scratch = os.path.join('/scratch', USER, PBS_JOBID)
    src = os.getenv('PBS_O_WORKDIR')

    path_pattern = '/home/dave/SimResults/NREL5MW/D0022/prepost-data/*.xz'

    for ffname in tqdm(glob.glob(path_pattern)):
        appendix = os.path.basename(ffname).split('_')[0]
        with tarfile.open(ffname, mode='r:xz') as tar:
            # create new tar files if necessary for each [Case folder]
            for tarinfo in tar.getmembers():
                t2_name = os.path.basename(os.path.dirname(tarinfo.name))
                t2_dir = os.path.join(os.path.dirname(path_pattern), t2_name)
                if not os.path.isdir(t2_dir):
                    os.makedirs(t2_dir)
                t2_path = os.path.join(t2_dir, t2_name + '_%s.tar' % appendix)
                fileobj = tar.extractfile(tarinfo)
                # change the location of the file in the new archive:
                # the location of the archive is according to the folder
                # structure as defined in the tags, remove any subfolders
                tarinfo.name = os.basename(tarinfo.name)
                with tarfile.open(t2_path, mode='a') as t2:
                    t2.addfile(tarinfo, fileobj)


def merge_from_tarfiles(df_fname, path, pattern, tarmode='r:xz', tqdm=False,
                        header='infer', names=None, sep=',', min_itemsize={},
                        verbose=False, dtypes={}):
    """Merge all csv files from various tar archives into a big pd.DataFrame
    store.

    Parameters
    ----------

    df_fname : str
        file name of the pd.DataFrame h5 store in which all chunks will be
        merged. Names usually used are:
            * [sim_id]_ErrorLogs.h5
            * [sim_id]_statistics.h5

    path : str
        Directory in which all chunks are located.

    pattern : str
        Search pattern used to select (using glob.glob) files in path

    tarmode : str, default='r:xz'
        File opening mode for tarfile (used when opening each of the chunks).

    tqdm : boolean, default=False
       If True, an interactive progress bar will be displayed (requires the
       tqdm module). If set to False no progress bar will be displayed.

    header : int, default='infer'
        Argument passed on to pandas.read_csv. Default to 'infer', set to
        None if there is no header, set to 0 if header is on first row.

    names : list of column names, default=None
        Argument passed on to pandas.read_csv. Default to None. List with
        column names to be used in the DataFrame.

    min_itemsize : dict, default={}
        Argument passed on to pandas.HDFStore.append. Set the minimum lenght
        for a given column in the DataFrame.

    sep : str, default=','
        Argument passed on to pandas.read_csv. Set to ';' when handling the
        ErrorLogs.

    """

    store = pd.HDFStore(os.path.join(path, df_fname), mode='w', format='table',
                        complevel=9, complib='zlib')

    if tqdm:
        from tqdm import tqdm
    else:
        def tqdm(itereable):
            return itereable

    if len(glob.glob(os.path.join(path, pattern))) < 1:
        raise RuntimeError('No files found to merge')

    for tar_fname in tqdm(glob.glob(os.path.join(path, pattern))):
        if verbose:
            print(tar_fname)
        with tarfile.open(tar_fname, mode=tarmode) as tar:
            df = pd.DataFrame()
            for tarinfo in tar.getmembers():
                fileobj = tar.extractfile(tarinfo)
                tmp = pd.read_csv(fileobj, header=header, names=names, sep=sep)
                for col, dtype in dtypes.items():
                    tmp[col] = tmp[col].astype(dtype)
                df = df.append(tmp)
            try:
                if verbose:
                    print('writing...')
                store.append('table', df, min_itemsize=min_itemsize)
            except Exception as e:
                if verbose:
                    print('store columns:')
                    print(store.select('table', start=0, stop=0).columns)
                    print('columns of the DataFrame being added:')
                    print(df.columns)
                storecols = store.select('table', start=0, stop=0).columns
                store.close()
                print(e)
                return df, storecols

    store.close()


# TODO: make this class more general so you can also just give a list of files
# to be merged, excluding the tar archives.
class AppendDataFrames(object):
    """Merge DataFrames, either in h5 or csv format, located in (compressed)
    tar archives.
    """

    def __init__(self, tqdm=False):
        if tqdm:
            from tqdm import tqdm
        else:
            def tqdm(itereable):
                return itereable
        self.tqdm = tqdm

    def _open(self, fname, tarmode='r:xz'):
        """Open text file directly or from a tar archive. Return iterable
        since a tar archive might contain several csv text files
        """

        if fname.find('.tar') > -1:
            with tarfile.open(fname, mode=tarmode) as tar:
                for tarinfo in tar.getmembers():
                    linesb = tar.extractfile(tarinfo).readlines()
                    # convert from bytes to strings
                    lines = [line.decode() for line in linesb]
                    yield lines, tarinfo.name
        else:
            with open(fname, 'r') as f:
                lines = f.readlines()
            yield lines, os.path.basename(fname)

    def df2store(self, store, path, tarmode='r:xz', min_itemsize={},
                 colnames=None, header='infer', columns=None, sep=';',
                 index2col=None, ignore_index=True, fname_col=False):
        """This is very slow, use txt2txt instead.
        """

        # TODO: it seems that with threading you could parallelize this kind
        # of work: http://stackoverflow.com/q/23598063/3156685
        # http://stackoverflow.com/questions/23598063/
        # multithreaded-web-scraper-to-store-values-to-pandas-dataframe

        # http://gouthamanbalaraman.com/blog/distributed-processing-pandas.html

        df = pd.DataFrame()
        for fname in self.tqdm(glob.glob(path)):
            with tarfile.open(fname, mode=tarmode) as tar:
                df = pd.DataFrame()
                for tarinfo in tar.getmembers():
                    fileobj = tar.extractfile(tarinfo)
                    if tarinfo.name[-2:] == 'h5':
                        tmp = pd.read_hdf(fileobj, 'table', columns=columns)
                    elif tarinfo.name[-3:] == 'csv':
                        tmp = pd.read_csv(fileobj, sep=sep, names=colnames,
                                          header=header, usecols=columns)
                    else:
                        continue
                    if index2col is not None:
                        # if the index does not have a name we can still set it
                        tmp[index2col] = tmp.index
                        tmp[index2col] = tmp[index2col].astype(str)
                        tmp.reset_index(level=0, drop=True, inplace=True)
                    # add the file name as a column
                    if fname_col:
                        case_id = os.path.basename(tarinfo.name)
                        tmp[fname_col] = '.'.join(case_id.split('.')[:-1])
                        tmp[fname_col] = tmp[fname_col].astype(str)
                    df = df.append(tmp, ignore_index=ignore_index)

                store.append('table', df, min_itemsize=min_itemsize)
#                if len(df) > w_every:
#                    # and merge into the big ass DataFrame
#                    store.append('table', df, min_itemsize=min_itemsize)
#                    df = pd.DataFrame()
        return store

    # FIXME: when merging log file analysis (files with header), we are still
    # skipping over one case
    def txt2txt(self, fjoined, path, tarmode='r:xz', header=None, sep=';',
                fname_col=False, header_fjoined=None, recursive=False):
        """Read as strings, write to another file as strings.

        Parameters
        ----------

        fjoined

        path

        tarmode

        header : int, default=None
            Indicate if data files contain a header and on which line it is
            located. Set to None if data files do not contain header, and in
            that case the joined file will not contain a header either. All
            lines above the header are ignored.

        sep

        fname_col

        header_fjoined : str, default=None
            If the data files do not contain a header write out header_fjoined
            as the header of the joined file.

        recursive

        Return
        ------

        header_fjoined : str
            String of the header that was written to the joined file.

        """
        if isinstance(header, int):
            write_header = True
            icut = header + 1
        else:
            # when header is None, there is no header
            icut = 0
            write_header = False
        if isinstance(header_fjoined, str):
            write_header = True

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as ft:
            ftname = ft.name
            for fname in self.tqdm(glob.glob(path, recursive=recursive)):
                for lines, case_id in self._open(fname, tarmode=tarmode):
                    # only include the header at the first round
                    if write_header:
                        if header_fjoined is None:
                            header_fjoined = lines[header]
                        # add extra column with the file name if applicable
                        if fname_col:
                            rpl = sep + fname_col + '\n'
                            header_fjoined = header_fjoined.replace('\n', rpl)
                        ft.write(header_fjoined)
                        write_header = False
                    # but cut out the header on all other occurances
                    case_id = '.'.join(case_id.split('.')[:-1])
                    for line in lines[icut:]:
                        if fname_col:
                            line = line.replace('\n', sep + case_id + '\n')
                        ft.write(line)
                ft.flush()

        # and move from temp dir to fjoined
        shutil.move(ftname, fjoined)
        return header_fjoined.replace('\n', '')

    def csv2df_chunks(self, store, fcsv, chunksize=100000, min_itemsize={},
                      colnames=None, dtypes={}, header='infer', sep=';'):
        """Convert a large csv file to a pandas.DataFrame in chunks using
        a pandas.HDFStore.
        """
        df_iter = pd.read_csv(fcsv, chunksize=chunksize, sep=sep,
                              names=colnames, header=header)
        for df_chunk in self.tqdm(df_iter):
            for col, dtype in dtypes.items():
                df_chunk[col] = df_chunk[col].astype(dtype)
            store.append('table', df_chunk, min_itemsize=min_itemsize)
        return store


if __name__ == '__main__':
    pass
