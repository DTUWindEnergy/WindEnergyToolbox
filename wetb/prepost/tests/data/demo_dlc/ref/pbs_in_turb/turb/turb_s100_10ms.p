
### Standard Output
#PBS -N no_name_job
#PBS -o ./pbs_out/turb/turb_s100_10ms.out
### Standard Error
#PBS -e ./pbs_out/turb/turb_s100_10ms.err
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=00:59:59
#PBS -lnodes=1:ppn=1
### Queue name
#PBS -q workq

### #PBS -a [start_time]
### #PBS -W depend=afterany:[job_id]

### Browse to current working dir
echo ""
cd $PBS_O_WORKDIR
echo "current working dir:"
pwd
echo ""

### ===========================================================================
echo "------------------------------------------------------------------------"
echo "PRELUDE"
echo "------------------------------------------------------------------------"

winefix
cd /scratch/$USER/$PBS_JOBID/


echo ""
echo "------------------------------------------------------------------------"
echo "EXECUTION"
echo "------------------------------------------------------------------------"

time WINEARCH=win64 WINEPREFIX=~/.wine wine mann_turb_x64.exe 'turb_s100_10ms' 1.000000 29.400000 3.000000 100 8192 32 32 0.7812 6.5000 6.5000 1
### wait for jobs to finish
wait

echo ""
echo "------------------------------------------------------------------------"
echo "CODA"
echo "------------------------------------------------------------------------"

# COPY BACK FROM SCRATCH AND RENAME, remove _ at end
cp 'turb_s100_10ms_u.bin' "$PBS_O_WORKDIR/../turb/turb_s100_10msu.bin"
cp 'turb_s100_10ms_v.bin' "$PBS_O_WORKDIR/../turb/turb_s100_10msv.bin"
cp 'turb_s100_10ms_w.bin' "$PBS_O_WORKDIR/../turb/turb_s100_10msw.bin"


echo ""
### ===========================================================================
exit
