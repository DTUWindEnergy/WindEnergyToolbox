### Standard Output
#PBS -N dlc01_steady_wsp11_s101
#PBS -o 'pbs_out/dlc01_demos/dlc01_steady_wsp11_s101.out'
### Standard Error
#PBS -e 'pbs_out/dlc01_demos/dlc01_steady_wsp11_s101.err'
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=1
### Queue name
#PBS -q workq

# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  ### Create scratch directory and copy data to it
  cd "$PBS_O_WORKDIR"
  echo 'current working dir (pwd):'
  pwd
  cp -R 'demo_dlc_remote.zip' "/scratch/$USER/$PBS_JOBID"
fi
# ==============================================================================


# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  echo
  echo 'Execute commands on scratch nodes'
  cd "/scratch/$USER/$PBS_JOBID"
  # create unique dir for each CPU
  mkdir '1'; cd '1'
  pwd
  /usr/bin/unzip ../demo_dlc_remote.zip
  mkdir -p 'htc/dlc01_demos/'
  mkdir -p 'res/dlc01_demos/'
  mkdir -p 'logfiles/dlc01_demos/'
  mkdir -p 'turb/'
  mkdir -p 'turb_micro/'
  mkdir -p 'turb_meander/'
  cp -R "$PBS_O_WORKDIR/htc/dlc01_demos/dlc01_steady_wsp11_s101.htc" 'htc/dlc01_demos/'
  cp -R "$PBS_O_WORKDIR/../turb/turb_s101_11ms"*.bin "turb/"
  cp -R "$PBS_O_WORKDIR/turb_micro/micro_s101_11ms"*.bin "turb_micro/"
  cp -R "$PBS_O_WORKDIR/turb_meander/meander_s101_11ms"*.bin "turb_meander/"
  _HOSTNAME_=`hostname`
  if [[ ${_HOSTNAME_:0:1} == "j" ]] ; then
    WINEARCH='win32' WINEPREFIX="$HOME/.wine32" winefix
  fi
# ==============================================================================

# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  # with find+xargs we first browse to CPU folder
  cd "$CPU_NR"
fi
# ------------------------------------------------------------------------------

echo ''
# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  echo 'execute HAWC2, fork to background'
  time WINEARCH='win32' WINEPREFIX="$HOME/.wine32" wine 'hawc2-latest' 'htc/dlc01_demos/dlc01_steady_wsp11_s101.htc' &
  wait
# ==============================================================================

# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  echo 'execute HAWC2, do not fork and wait'
  (time WINEARCH='win32' WINEPREFIX="$HOME/.wine32" numactl --physcpubind=$CPU_NR wine 'hawc2-latest' 'htc/dlc01_demos/dlc01_steady_wsp11_s101.htc') 2>&1 | tee 'pbs_out/dlc01_demos/dlc01_steady_wsp11_s101.err.out'
fi
# ------------------------------------------------------------------------------


### Epilogue
# ==============================================================================
# single PBS mode: one case per PBS job
# evaluates to true if LAUNCH_PBS_MODE is NOT set
if [ -z ${LAUNCH_PBS_MODE+x} ] ; then
  ### wait for jobs to finish
  wait
  echo ''
  echo 'Copy back from scratch directory'
  cd "/scratch/$USER/$PBS_JOBID/1/"
  mkdir -p "$PBS_O_WORKDIR/res/dlc01_demos/"
  mkdir -p "$PBS_O_WORKDIR/logfiles/dlc01_demos/"
  mkdir -p "$PBS_O_WORKDIR/animation/"
  mkdir -p "$PBS_O_WORKDIR/../turb/"
  mkdir -p "$PBS_O_WORKDIR/turb_micro/"
  mkdir -p "$PBS_O_WORKDIR/turb_meander/"
  cp -R 'res/dlc01_demos/.' "$PBS_O_WORKDIR/res/dlc01_demos/."
  cp -R 'logfiles/dlc01_demos/.' "$PBS_O_WORKDIR/logfiles/dlc01_demos/."
  cp -R 'animation/.' "$PBS_O_WORKDIR/animation/."

  echo ''
  echo 'COPY BACK TURB IF APPLICABLE'
  cd 'turb/'
  for i in `ls *.bin`; do  if [ -e "$PBS_O_WORKDIR/../turb/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "$PBS_O_WORKDIR/../turb/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/1/"
  cd 'turb_micro/'
  for i in `ls *.bin`; do  if [ -e "$PBS_O_WORKDIR/turb_micro/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "$PBS_O_WORKDIR/turb_micro/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/1/"
  cd 'turb_meander/'
  for i in `ls *.bin`; do  if [ -e "$PBS_O_WORKDIR/turb_meander/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "$PBS_O_WORKDIR/turb_meander/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/1/"
  echo 'END COPY BACK TURB'
  echo ''

  echo 'COPYBACK [copyback_files]/[copyback_frename]'
  echo 'END COPYBACK'
  echo ''
  echo ''
  echo 'following files are on node/cpu 1 (find .):'
  find .
# ==============================================================================
# ------------------------------------------------------------------------------
# find+xargs mode: 1 PBS job, multiple cases
else
  cd "/scratch/$USER/$PBS_JOBID/$CPU_NR/"
  rsync -a --remove-source-files 'res/dlc01_demos/.' "../remote/res/dlc01_demos/."
  rsync -a --remove-source-files 'logfiles/dlc01_demos/.' "../remote/logfiles/dlc01_demos/."
  rsync -a --remove-source-files 'pbs_out/dlc01_demos/.' "../remote/pbs_out/dlc01_demos/."
  rsync -a --remove-source-files 'animation/.' "../remote/animation/."

  echo ''
  echo 'COPY BACK TURB IF APPLICABLE'
  cd 'turb/'
  for i in `ls *.bin`; do  if [ -e "$PBS_O_WORKDIR/../turb/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "$PBS_O_WORKDIR/../turb/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/$CPU_NR/"
  cd 'turb_micro/'
  for i in `ls *.bin`; do  if [ -e "../remote/turb_micro/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "../remote/turb_micro/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/$CPU_NR/"
  cd 'turb_meander/'
  for i in `ls *.bin`; do  if [ -e "../remote/turb_meander/$i" ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp "$i" "../remote/turb_meander/"; fi; done
  cd "/scratch/$USER/$PBS_JOBID/$CPU_NR/"
  echo 'END COPY BACK TURB'
  echo ''

  echo 'COPYBACK [copyback_files]/[copyback_frename]'
  echo 'END COPYBACK'
  echo ''
# ------------------------------------------------------------------------------
fi
exit
