from wetb.hawc2.htc_file import HTCFile
import os
from os import path

pbs_template = '''### Standard Output
#PBS -N [jobname]
#PBS -o [pbsoutdir]/[jobname].out
### Standard Error
#PBS -e [pbsoutdir]/[jobname].err
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -l nodes=1:ppn=1
### Queue name
#PBS -q workq

# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  ### Create scratch directory and copy data to it
  cd $PBS_O_WORKDIR
  echo "current working dir (pwd):"
  pwd
  cp -R ./[modelzip] /scratch/$USER/$PBS_JOBID
fi
# ==============================================================================


# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  echo
  echo 'Execute commands on scratch nodes'
  cd /scratch/$USER/$PBS_JOBID
  # create unique dir for each CPU
  mkdir "1"; cd "1"
  pwd
  /usr/bin/unzip ../[modelzip]
  mkdir -p [htcdir]
  mkdir -p [resdir]
  mkdir -p [logdir]
  mkdir -p [turbdir]
  cp -R $PBS_O_WORKDIR/[htcdir]/[jobname].htc ./[htcdir]
  cp -R $PBS_O_WORKDIR/[turbdir][turbfileroot]*.bin [turbdir]
  _HOSTNAME_=`hostname`
  if [[ ${_HOSTNAME_:0:1} == "j" ]] ; then
    WINEARCH=win64 WINEPREFIX=~/.wine winefix
  fi
# ==============================================================================

# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  # with find+xargs we first browse to CPU folder
  cd "$CPU_NR"
fi
# ------------------------------------------------------------------------------

echo ""
# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  echo "execute HAWC2, fork to background"
  time WINEARCH=win64 WINEPREFIX=~/.wine wine hawc2-latest ./[htcdir]/[jobname].htc &
  wait
# ==============================================================================

# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  echo "execute HAWC2, do not fork and wait"
  (time WINEARCH=win64 WINEPREFIX=~/.wine numactl --physcpubind=$CPU_NR wine hawc2-latest ./[htcdir]/[jobname].htc) 2>&1 | tee [pbsoutdir]/[jobname].err.out
fi
# ------------------------------------------------------------------------------


### Epilogue
# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  ### wait for jobs to finish
  wait
  echo ""
  echo "Copy back from scratch directory"
  mkdir -p $PBS_O_WORKDIR/[resdir]
  mkdir -p $PBS_O_WORKDIR/[logdir]
  mkdir -p $PBS_O_WORKDIR/animation/
  mkdir -p $PBS_O_WORKDIR/[turbdir]
  cp -R [resdir]. $PBS_O_WORKDIR/[resdir].
  cp -R [logdir]. $PBS_O_WORKDIR/[logdir].
  cp -R animation/. $PBS_O_WORKDIR/animation/.

  echo ""
  echo "COPY BACK TURB IF APPLICABLE"
  cd turb/
  for i in `ls *.bin`; do  if [ -e $PBS_O_WORKDIR/[turbdir]$i ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp $i $PBS_O_WORKDIR/[turbdir]; fi; done
  cd /scratch/$USER/$PBS_JOBID/1/
  echo "END COPY BACK TURB"
  echo ""

  echo "COPYBACK [copyback_files]/[copyback_frename]"
  echo "END COPYBACK"
  echo ""
  echo ""
  echo "following files are on node/cpu 1 (find .):"
  find .
# ==============================================================================
# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  cd /scratch/$USER/$PBS_JOBID/$CPU_NR/
  rsync -a --remove-source-files [resdir]. ../HAWC2SIM/[resdir].
  rsync -a --remove-source-files [logdir]. ../HAWC2SIM/[logdir].
  rsync -a --remove-source-files [pbsoutdir]/. ../HAWC2SIM/[pbsoutdir]/.
  rsync -a --remove-source-files animation/. ../HAWC2SIM/animation/.

  echo ""
  echo "COPY BACK TURB IF APPLICABLE"
  cd turb/
  for i in `ls *.bin`; do  if [ -e $PBS_O_WORKDIR/[turbdir]$i ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp $i $PBS_O_WORKDIR/[turbdir]; fi; done
  cd /scratch/$USER/$PBS_JOBID/$CPU_NR/
  echo "END COPY BACK TURB"
  echo ""

  echo "COPYBACK [copyback_files]/[copyback_frename]"
  echo "END COPYBACK"
  echo ""
# ------------------------------------------------------------------------------
fi
exit
'''

def htc2pbs(htc_fn, walltime='00:40:00', zipfile=None):
    """
    Creates a PBS launch file (.p) based on a HAWC2 .htc file.
    - Assumes htc files are within a htc/[casename]/ directory relative to current directory.
    - Assumes there is a .zip file in the current directory which contains the turbine model.
      If there is none, the zip file is set to 'model.zip' by default
    - Will place a .p fine in pbs_in/[casename]/ directory relative to current directory.
    -

    Parameters
    ----------
    htc_fn : str
        The file name and path to the .htc file relative to current directory.
    walltime: str (default='00:40:00')
        A string indicating the walltime of the job of the form 'HH:MM:SS'
    zipfile: str (default=None)
        The filename of the zipfile containing the wind turbine model files and
        HAWC2 executable. if zipfile=None, searches the current directory for a
        zip file. If none is found, sets zipfile to 'model.zip'


    Returns
    -------
    str
        The filename and path to the output .p file

    Raises
    ------
    FileNotFoundError: If the file structure is not correct.
    """



    basename = path.relpath(path.dirname(htc_fn), 'htc')
    jobname = path.splitext(path.basename(htc_fn))[0]
    pbs_in_dir = path.join('pbs_in', basename)
    if basename == '.':
        raise FileNotFoundError('File structure is incorrect.')

    if not zipfile:
        try:
            zipfile = [x for x in os.listdir() if x.lower().endswith('.zip')][0]
        except:
            print('No .zip file found in current directory. Set model zip to \'model.zip\'')
            zipfile = 'model.zip'

    #   get the required parameters for the pbs file from the htc file
    htc = HTCFile(htc_fn) #modelpath='../..')
    p = {
        'walltime'      : walltime,
        'modelzip'      : zipfile,
        'jobname'       : jobname,
        'htcdir'        : 'htc/' + basename,
        'logdir'        :  path.dirname(htc.simulation.logfile.str_values())[2:] + '/',
        'resdir'        : path.dirname(htc.output.filename.str_values())[2:] + '/',
        'turbdir'       : path.dirname(htc.wind.mann.filename_u.str_values()) + '/',
        'turbfileroot'  : path.basename(htc.wind.mann.filename_u.str_values()).split('u.')[0],
        'pbsoutdir'     : 'pbs_out/' + basename
        }


    #Write pbs file based on template file and tags
    if not os.path.exists(pbs_in_dir):
        os.makedirs(pbs_in_dir)

    template = str(pbs_template)

    for key, value in p.items():
        template = template.replace('[' + key + ']', value)

    with open(os.path.join(pbs_in_dir, jobname + '.p'), 'w') as f:
        f.write(template)
