### Standard Output 
#PBS -N dlc01_steady_wsp8_noturb 
#PBS -o ./pbs_out/dlc01_demos/dlc01_steady_wsp8_noturb.out
### Standard Error 
#PBS -e ./pbs_out/dlc01_demos/dlc01_steady_wsp8_noturb.err
#PBS -W umask=003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=1
### Queue name
#PBS -q workq
### Create scratch directory and copy data to it 
cd $PBS_O_WORKDIR
echo "current working dir (pwd):"
pwd 
cp -R ./demo_dlc_remote.zip /scratch/$USER/$PBS_JOBID


echo ""
echo "Execute commands on scratch nodes"
cd /scratch/$USER/$PBS_JOBID
pwd
/usr/bin/unzip demo_dlc_remote.zip
mkdir -p htc/dlc01_demos/
mkdir -p res/dlc01_demos/
mkdir -p logfiles/dlc01_demos/
mkdir -p turb/
cp -R $PBS_O_WORKDIR/htc/dlc01_demos/dlc01_steady_wsp8_noturb.htc ./htc/dlc01_demos/
cp -R $PBS_O_WORKDIR/../turb/none*.bin turb/ 
time WINEARCH=win32 WINEPREFIX=~/.wine32 wine hawc2-latest ./htc/dlc01_demos/dlc01_steady_wsp8_noturb.htc  &
### wait for jobs to finish 
wait
echo ""
echo "Copy back from scratch directory" 
cd /scratch/$USER/$PBS_JOBID
mkdir -p $PBS_O_WORKDIR/res/dlc01_demos/
mkdir -p $PBS_O_WORKDIR/logfiles/dlc01_demos/
mkdir -p $PBS_O_WORKDIR/animation/
mkdir -p $PBS_O_WORKDIR/../turb/
cp -R res/dlc01_demos/. $PBS_O_WORKDIR/res/dlc01_demos/.
cp -R logfiles/dlc01_demos/. $PBS_O_WORKDIR/logfiles/dlc01_demos/.
cp -R animation/. $PBS_O_WORKDIR/animation/.

echo ""
echo "COPY BACK TURB IF APPLICABLE"
cd turb/
for i in `ls *.bin`; do  if [ -e $PBS_O_WORKDIR/../turb/$i ]; then echo "$i exists no copyback"; else echo "$i copyback"; cp $i $PBS_O_WORKDIR/../turb/; fi; done
cd /scratch/$USER/$PBS_JOBID
echo "END COPY BACK TURB"
echo ""

echo ""
echo "following files are on the node (find .):"
find .
exit
