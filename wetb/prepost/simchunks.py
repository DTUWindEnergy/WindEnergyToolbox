# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:22:49 2016

@author: dave
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import dict
from io import open
from builtins import zip
from builtins import range
from builtins import str
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import object

# standard python library
import os
import zipfile
import copy
import tarfile
import glob

import numpy as np
import pandas as pd
#from tqdm import tqdm

from wetb.prepost.Simulations import Cases


def create_chunks_htc_pbs(cases, sort_by_values=['[Windspeed]'], ppn=20,
                          nr_procs_series=9, processes=1, queue='workq',
                          walltime='24:00:00', chunks_dir='zip-chunks-jess',
                          pyenv='wetb_py3', i0=0):
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
                zf.writestr(n, zf_model.open(n).read())

        # create all necessary directories in the zip file
        dirtags = ['[htc_dir]', '[res_dir]','[log_dir]','[animation_dir]',
                   '[pbs_in_dir]', '[eigenfreq_dir]','[turb_dir]','[wake_dir]',
                   '[meander_dir]','[hydro_dir]', '[mooring_dir]',
                   '[pbs_in_dir]', '[pbs_out_dir]']
        dirnames = []
        for tag in dirtags:
            for dirname in set(df[tag].unique().tolist()):
                if not dirname or dirname.lower() not in ['false', 'none', 0]:
                    dirnames.append(dirname)
        for dirname in set(dirnames):
            if dirname != 0:
                zf.write('.', os.path.join(dirname, '.'))

        # and the post-processing data
        # FIXME: do not use hard coded paths!
        zf.write('.', 'prepost-data/')

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
            zf.write(src+'.htc', dst_rel+'.htc')

        # PBS files
        df_src = df['[run_dir]'] + df['[pbs_in_dir]'] + df['[case_id]']
        df_dst = df['[pbs_in_dir]'] + df['[case_id]']
        # Since df_src and df_dst are already Series, iterating is fast an it
        # is slower to first convert to a list
        for src, dst_rel in zip(df_src, df_dst):
            zf.write(src+'.p', dst_rel+'.p')

        # copy and rename input files with given versioned name to the
        # all files that will have to be renamed to their non-changeable
        # default file name.
        # this is a bit more tricky since unique() will not work on list items
        copyto_files_tmp = df['[copyto_files]'].astype(str)
        copyto_files = []
        # cycle through the unique elements
        for k in set(copyto_files_tmp):
            # k is of form: "['some/file.txt', 'another/file1.txt']"
            if len(k) < 2:
                continue
            items = [kk[1:-1] for kk in k.split('[')[1].split(']')[0].split(', ')]
            copyto_files.extend(items)
        # we might still have non unique elements
        copyto_files = set(copyto_files)
        for copyto_file, dst_rel in zip(copyto_files, df_dst):
            src = os.path.join(run_dir, copyto_file)
            # make dir if it does not exist
            zf.write('.', os.path.dirname(copyto_file), '.')
            zf.write(src, copyto_file)

        zf.close()

        return fname, df_index

    pbs_tmplate ="""
### Standard Output
#PBS -N [job_name]
#PBS -o [std_out]
### Standard Error
#PBS -e [std_err]
#PBS -W umask=[umask]
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -l nodes=[nodes]:ppn=[ppn]
### Queue name
#PBS -q [queue]

"""

    def make_pbs_chunks(df, ii, sim_id, run_dir, model_zip):
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

        pbase = os.path.join('/scratch','$USER', '$PBS_JOBID', '')
        post_dir_base = post_dir.split(sim_id)[1]
        if post_dir_base[0] == os.path.sep:
            post_dir_base = post_dir_base[1:]

        pbs_in_base = os.path.commonpath(df['[pbs_in_dir]'].unique().tolist())
        pbs_in_base = os.path.join(pbs_in_base, '')
        htc_base = os.path.commonpath(df['[htc_dir]'].unique().tolist())
        htc_base = os.path.join(htc_base, '')
        res_base = os.path.commonpath(df['[res_dir]'].unique().tolist())
        res_base = os.path.join(res_base, '')
        log_base = os.path.commonpath(df['[log_dir]'].unique().tolist())
        log_base = os.path.join(log_base, '')

        # =====================================================================
        # PBS HEADER
        pbs = copy.copy(pbs_tmplate)
        pbs = pbs.replace('[job_name]', jobid)
        pbs = pbs.replace('[std_out]', './pbs_out_chunks/%s.out' % jobid)
        pbs = pbs.replace('[std_err]', './pbs_out_chunks/%s.err' % jobid)
        pbs = pbs.replace('[umask]', '0003')
        pbs = pbs.replace('[walltime]', walltime)
        pbs = pbs.replace('[nodes]', str(nodes))
        pbs = pbs.replace('[ppn]', str(ppn))
        pbs = pbs.replace('[queue]', queue)
        pbs += '\necho "%s"\n' % ('-'*70)

        # =====================================================================
        # activate the python environment
        pbs += 'echo "activate python environment %s"\n' % pyenv
        pbs += 'source activate %s\n' % pyenv
        # sometimes activating an environment fails due to a FileExistsError
        # is this because it is activated at the same time on another node?
        # check twice if the environment got activated for real
        pbs += 'echo "CHECK 2x IF %s IS ACTIVE, IF NOT TRY AGAIN"\n' % pyenv
        pbs += 'CMD=\"from distutils.sysconfig import get_python_lib;'
        pbs += 'print (get_python_lib().find(\'%s\'))"\n' % pyenv
        pbs += 'ACTIVATED=`python -c "$CMD"`\n'
        pbs += 'if [ $ACTIVATED -eq -1 ]; then source activate %s;fi\n' % pyenv
        pbs += 'ACTIVATED=`python -c "$CMD"`\n'
        pbs += 'if [ $ACTIVATED -eq -1 ]; then source activate %s;fi\n' % pyenv

        # =====================================================================
        # create all necessary directories at CPU_NR dirs, turb db dirs, sim_id
        # browse to scratch directory
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n\n'
        pbs += 'echo "create CPU directories on the scratch disk"\n'
        pbs += 'mkdir -p %s\n' % os.path.join(pbase, sim_id, '')
        for k in range(ppn):
            pbs += 'mkdir -p %s\n' % os.path.join(pbase, '%i' % k, '')
        # pretend to be on the scratch sim_id directory to maintain the same
        # database turb structure
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % os.path.join(pbase, sim_id, '')
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "create turb_db directories"\n'
        db_dir_tags = ['[turb_db_dir]', '[meand_db_dir]', '[wake_db_dir]']
        turb_dirs = []
        for tag in db_dir_tags:
            for dirname in set(df[tag].unique().tolist()):
                if not dirname or dirname.lower() not in ['false', 'none']:
                    turb_dirs.append(dirname)
        turb_dirs = set(turb_dirs)
        for dirname in turb_dirs:
            pbs += 'mkdir -p %s\n' % os.path.join(dirname, '')

        # =====================================================================
        # get the zip-chunk file from the PBS_O_WORKDIR
        pbs += '\n'
        pbs += 'echo "%s"\n' % ('-'*70)
        pbs += 'cd $PBS_O_WORKDIR\n'
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "get the zip-chunk file from the PBS_O_WORKDIR"\n'
        # copy the relevant zip chunk file to the scratch main directory
        rpl = (os.path.join('./', chunks_dir, jobid), os.path.join(pbase, ''))
        pbs += 'cp %s.zip %s\n' % rpl

        # turb_db_dir might not be set, same for turb_base_name, for those
        # cases we do not need to copy anything from the database to the node
        base_name_tags = ['[turb_base_name]', '[meand_base_name]',
                          '[wake_base_name]']
        for db, base_name in zip(db_dir_tags, base_name_tags):
            turb_db_dirs = df[db] + df[base_name]
            # When set to None, the DataFrame will have text as None
            turb_db_src = turb_db_dirs[turb_db_dirs.str.find('None')==-1]
            pbs += '\n'
            pbs += '# copy to scratch db directory for %s, %s\n' % (db, base_name)
            for k in turb_db_src.unique():
                dst = os.path.dirname(os.path.join(pbase, sim_id, k))
                pbs += 'cp %s* %s\n' % (k, os.path.join(dst, '.'))

        # =====================================================================
        # browse back to the scratch directory
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "unzip chunk, create dirs in cpu and sim_id folders"\n'
        # unzip chunk, this contains all relevant folders already, and also
        # contains files defined in [copyto_files]
        for k in list(range(ppn)) + [sim_id]:
            dst = os.path.join('%s' % k, '.')
            pbs += '/usr/bin/unzip %s -d %s >> /dev/null\n' % (jobid+'.zip', dst)

        # create hard links for all the turbulence files
        turb_dir_base = os.path.join(os.path.commonpath(list(turb_dirs)), '')
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "copy all turb files into CPU dirs"\n'
        for k in range(ppn):
            rpl = (os.path.relpath(os.path.join(sim_id, turb_dir_base)), k)
            pbs += 'find %s -iname *.bin -exec cp {} %s/{} \\;\n' % rpl

        # =====================================================================
        # finally we can run find+xargs!!!
        pbs += '\n'
        pbs += 'echo "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "START RUNNING JOBS IN find+xargs MODE"\n'
        pbs += 'WINEARCH=win32 WINEPREFIX=~/.wine32 winefix\n'
        pbs += '# run all the PBS *.p files in find+xargs mode\n'
        pbs += 'echo "following cases will be run from following path:"\n'
        pbs += 'echo "%s"\n' % (os.path.join(sim_id, pbs_in_base))
        pbs += 'export LAUNCH_PBS_MODE=false\n'
        rpl = (cmd_find, os.path.join(sim_id, pbs_in_base))
        pbs += "%s %s -type f -name '*.p' | sort -z\n" % rpl
        pbs += '\n'
        pbs += 'echo "number of files to be launched: "'
        pbs += '`find %s -type f | wc -l`\n' % os.path.join(sim_id, pbs_in_base)
        rpl = (cmd_find, os.path.join(sim_id, pbs_in_base), cmd_xargs, ppn)
        cmd = ("%s %s -type f -name '*.p' -print0 | sort -z | %s -0 -I{} "
               "--process-slot-var=CPU_NR -n 1 -P %i sh {}\n" % rpl)
        pbs += cmd
        pbs += 'echo "END OF JOBS IN find+xargs MODE"\n'

        # =====================================================================
        # move results back from the node sim_id dir to the origin
        pbs += '\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += "echo 'total scratch disk usage:'\n"
        pbs += 'du -hs %s\n' % pbase
        pbs += 'cd %s\n' % os.path.join(pbase, sim_id)
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "Results saved at sim_id directory:"\n'
        rpl = (os.path.join(pbs_in_base, '*'), os.path.join(htc_base, '*'))
        pbs += 'find \n'

        # compress all result files into an archive, first *.sel files
        # FIXME: why doesn this work with -name "*.sel" -o -name "*.dat"??
        pbs += '\necho "move results into compressed archive"\n'
        pbs += 'find %s -name "*.sel" -print0 ' % res_base
        fname = os.path.join(res_base, 'resfiles_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname
        # now add the *.dat files to the archive
        pbs += 'find %s -name "*.dat" -print0 ' % res_base
        fname = os.path.join(res_base, 'resfiles_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname

        pbs += 'xz -z2 -T %i %s.tar\n' % (ppn, fname)

        # compress all logfiles into an archive
        pbs += '\necho "move logfiles into compressed archive"\n'
        pbs += 'find %s -name "*.log" -print0 ' % log_base
        fname = os.path.join(log_base, 'logfiles_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname
        pbs += 'xz -z2 -T %i %s.tar\n' % (ppn, fname)

        # compress all post-processing results (saved as csv's) into an archive
        pbs += '\necho "move statsdel into compressed archive"\n'
        pbs += 'find %s -name "*.csv" -print0 ' % res_base
        fname = os.path.join(post_dir_base, 'statsdel_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname
        pbs += 'xz -z2 -T %i %s.tar\n' % (ppn, fname)

        # compress all post-processing results (saved as csv's) into an archive
        pbs += '\necho "move log analysis into compressed archive"\n'
        pbs += 'find %s -name "*.csv" -print0 ' % log_base
        fname = os.path.join(post_dir_base, 'loganalysis_chnk_%05i' % ii)
        pbs += '| xargs -0 tar --remove-files -rf %s.tar\n' % fname
        pbs += 'xz -z2 -T %i %s.tar\n' % (ppn, fname)

        pbs += '\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += 'cd %s\n' % pbase
        pbs += "echo 'current working directory:'\n"
        pbs += 'pwd\n'
        pbs += 'echo "move results back from node scratch/sim_id to origin, '
        pbs += 'but ignore htc, and pbs_in directories."\n'

        tmp = os.path.join(sim_id, '*')
        pbs += 'echo "copy from %s to $PBS_O_WORKDIR/"\n' % tmp
        pbs += 'time rsync -au --remove-source-files %s $PBS_O_WORKDIR/ \\\n' % tmp
        pbs += '    --exclude %s \\\n' % os.path.join(pbs_in_base, '*')
        pbs += '    --exclude %s \n' % os.path.join(htc_base, '*')
        # when using -u, htc and pbs_in files should be ignored
#        pbs += 'time cp -ru %s $PBS_O_WORKDIR/\n' % tmp
        pbs += 'source deactivate\n'
        pbs += 'echo "DONE !!"\n'
        pbs += '\necho "%s"\n' % ('-'*70)
        pbs += '# in case wine has crashed, kill any remaining wine servers\n'
        pbs += '# caution: ALL the users wineservers will die on this node!\n'
        pbs += 'echo "following wineservers are still running:"\n'
        pbs += 'ps -u $USER -U $USER | grep wineserver\n'
        pbs += 'killall -u $USER wineserver\n'
        pbs += 'exit\n'

        rpl = (sim_id, ii)
        fname = os.path.join(run_dir, chunks_dir, '%s_chnk_%05i' % rpl)
        with open(fname+'.p', 'w') as f:
            f.write(pbs)

    def make_pbs_postpro_chunks():
        """When only the post-processing has to be re-done for a chunk.
        """
        pass


    cc = Cases(cases)
    df = cc.cases2df()
    # sort on the specified values in the given columns
    df.sort_values(by=sort_by_values, inplace=True)

    # create the directory to store all zipped chunks
    try:
        os.mkdir(os.path.join(df['[run_dir]'].iloc[0], chunks_dir))
    # FIXME: how do you make this work pythonically on both PY2 and PY3?
    except (FileExistsError, OSError):
        pass

    df_iter = chunker(df, nr_procs_series*ppn)
    sim_id = df['[sim_id]'].iloc[0]
    run_dir = df['[run_dir]'].iloc[0]
    model_zip = df['[model_zip]'].iloc[0]
    post_dir = df['[post_dir]'].iloc[0]
    nodes = 1
    df_ind = pd.DataFrame(columns=['chnk_nr'], dtype=np.int32)
    df_ind.index.name = '[case_id]'
    for ii, dfi in enumerate(df_iter):
        fname, ind = make_zip_chunks(dfi, i0+ii, sim_id, run_dir, model_zip)
        make_pbs_chunks(dfi, i0+ii, sim_id, run_dir, model_zip)
        df_ind = df_ind.append(ind)
        print(fname)

    fname = os.path.join(post_dir, 'case_id-chunk-index')
    df_ind['chnk_nr'] = df_ind['chnk_nr'].astype(np.int32)
    df_ind.to_hdf(fname+'.h5', 'table', compression=9, complib='zlib')
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
    return None, None


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

    def df2store(self, store, path, tarmode='r:xz', min_itemsize={},
                 colnames=None, header='infer', columns=None, sep=';',
                 index2col=None, ignore_index=True, fname_col=False):
        """
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
                fname_col=False):
        """Read as strings, write to another file as strings.
        """
        if header is not None:
            write_header = True
            icut = header + 1
        else:
            # when header is None, there is no header
            icut = 0
            write_header = False
        with open(fjoined, 'w') as f:
            for fname in self.tqdm(glob.glob(path)):
                with tarfile.open(fname, mode=tarmode) as tar:
                    for tarinfo in tar.getmembers():
                        linesb = tar.extractfile(tarinfo).readlines()
                        # convert from bytes to strings
                        lines = [line.decode() for line in linesb]
                        # only include the header at the first round
                        if write_header:
                            line = lines[header]
                            # add extra column with the file name if applicable
                            if fname_col:
                                rpl = sep + fname_col + '\n'
                                line = line.replace('\n', rpl)
                            f.write(line)
                            write_header = False
                        # but cut out the header on all other occurances
                        for line in lines[icut:]:
                            if fname_col:
                                case_id = os.path.basename(tarinfo.name)
                                case_id = '.'.join(case_id.split('.')[:-1])
                                line = line.replace('\n', sep + case_id + '\n')
                            f.write(line)
                f.flush()

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
