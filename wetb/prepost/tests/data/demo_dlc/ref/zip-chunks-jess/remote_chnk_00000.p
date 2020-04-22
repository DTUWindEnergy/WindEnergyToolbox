
### Standard Output
#PBS -N remote_chnk_00000
#PBS -o ./pbs_out_chunks/remote_chnk_00000.out
### Standard Error
#PBS -e ./pbs_out_chunks/remote_chnk_00000.err
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=09:00:00
#PBS -l nodes=1:ppn=20
### Queue name
#PBS -q workq


echo "----------------------------------------------------------------------"
echo "activate python environment py36-wetb"
source /home/python/miniconda3/bin/activate py36-wetb
echo "CHECK 2x IF py36-wetb IS ACTIVE, IF NOT TRY AGAIN"
CMD="from distutils.sysconfig import get_python_lib;print (get_python_lib().find('/usr/lib/python'))"
ACTIVATED=`python -c "$CMD"`
if [ $ACTIVATED -eq 0 ]; then source /home/python/miniconda3/bin/activate py36-wetb;fi
ACTIVATED=`python -c "$CMD"`
if [ $ACTIVATED -eq 0 ]; then source /home/python/miniconda3/bin/activate py36-wetb;fi

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/"
echo 'current working directory:'
pwd

echo "create CPU directories on the scratch disk"
mkdir -p "/scratch/$USER/$PBS_JOBID/remote/"
mkdir -p "/scratch/$USER/$PBS_JOBID/0/"
mkdir -p "/scratch/$USER/$PBS_JOBID/1/"
mkdir -p "/scratch/$USER/$PBS_JOBID/2/"
mkdir -p "/scratch/$USER/$PBS_JOBID/3/"
mkdir -p "/scratch/$USER/$PBS_JOBID/4/"
mkdir -p "/scratch/$USER/$PBS_JOBID/5/"
mkdir -p "/scratch/$USER/$PBS_JOBID/6/"
mkdir -p "/scratch/$USER/$PBS_JOBID/7/"
mkdir -p "/scratch/$USER/$PBS_JOBID/8/"
mkdir -p "/scratch/$USER/$PBS_JOBID/9/"
mkdir -p "/scratch/$USER/$PBS_JOBID/10/"
mkdir -p "/scratch/$USER/$PBS_JOBID/11/"
mkdir -p "/scratch/$USER/$PBS_JOBID/12/"
mkdir -p "/scratch/$USER/$PBS_JOBID/13/"
mkdir -p "/scratch/$USER/$PBS_JOBID/14/"
mkdir -p "/scratch/$USER/$PBS_JOBID/15/"
mkdir -p "/scratch/$USER/$PBS_JOBID/16/"

echo "----------------------------------------------------------------------"
cd $PBS_O_WORKDIR
echo 'current working directory:'
pwd
echo "get the zip-chunk file from the PBS_O_WORKDIR"
cp "zip-chunks-jess/remote_chnk_00000.zip" "/scratch/$USER/$PBS_JOBID/"

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/"
echo 'current working directory:'
pwd
echo "unzip chunk, create dirs in cpu and sim_id folders"
/usr/bin/unzip "remote_chnk_00000.zip" -d "0/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "1/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "2/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "3/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "4/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "5/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "6/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "7/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "8/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "9/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "10/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "11/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "12/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "13/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "14/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "15/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "16/." >> /dev/null
/usr/bin/unzip "remote_chnk_00000.zip" -d "remote/." >> /dev/null

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/remote/"
echo 'current working directory:'
pwd
echo "create turb_db directories"
mkdir -p "../turb/"

echo "----------------------------------------------------------------------"
cd "$PBS_O_WORKDIR"
echo 'current working directory:'
pwd

# copy to scratch db directory for [turb_db_dir], [turb_base_name]
cp "../turb/turb_s100_10ms"*.bin "/scratch/$USER/$PBS_JOBID/remote/../turb/."
cp "../turb/turb_s101_11ms"*.bin "/scratch/$USER/$PBS_JOBID/remote/../turb/."

# copy to scratch db directory for [meander_db_dir], [meander_base_name]

# copy to scratch db directory for [micro_db_dir], [micro_base_name]

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/"
echo 'current working directory:'
pwd
echo "create turb directories in CPU dirs"
mkdir -p "0/turb/"
mkdir -p "0/turb_meander/"
mkdir -p "0/turb_micro/"
mkdir -p "1/turb/"
mkdir -p "1/turb_meander/"
mkdir -p "1/turb_micro/"
mkdir -p "2/turb/"
mkdir -p "2/turb_meander/"
mkdir -p "2/turb_micro/"
mkdir -p "3/turb/"
mkdir -p "3/turb_meander/"
mkdir -p "3/turb_micro/"
mkdir -p "4/turb/"
mkdir -p "4/turb_meander/"
mkdir -p "4/turb_micro/"
mkdir -p "5/turb/"
mkdir -p "5/turb_meander/"
mkdir -p "5/turb_micro/"
mkdir -p "6/turb/"
mkdir -p "6/turb_meander/"
mkdir -p "6/turb_micro/"
mkdir -p "7/turb/"
mkdir -p "7/turb_meander/"
mkdir -p "7/turb_micro/"
mkdir -p "8/turb/"
mkdir -p "8/turb_meander/"
mkdir -p "8/turb_micro/"
mkdir -p "9/turb/"
mkdir -p "9/turb_meander/"
mkdir -p "9/turb_micro/"
mkdir -p "10/turb/"
mkdir -p "10/turb_meander/"
mkdir -p "10/turb_micro/"
mkdir -p "11/turb/"
mkdir -p "11/turb_meander/"
mkdir -p "11/turb_micro/"
mkdir -p "12/turb/"
mkdir -p "12/turb_meander/"
mkdir -p "12/turb_micro/"
mkdir -p "13/turb/"
mkdir -p "13/turb_meander/"
mkdir -p "13/turb_micro/"
mkdir -p "14/turb/"
mkdir -p "14/turb_meander/"
mkdir -p "14/turb_micro/"
mkdir -p "15/turb/"
mkdir -p "15/turb_meander/"
mkdir -p "15/turb_micro/"
mkdir -p "16/turb/"
mkdir -p "16/turb_meander/"
mkdir -p "16/turb_micro/"

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/remote/"
echo 'current working directory:'
pwd
echo "Link all turb files into CPU dirs"
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/0/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/1/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/2/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/3/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/4/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/5/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/6/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/7/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/8/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/9/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/10/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/11/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/12/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/13/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/14/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/15/turb/" \;
find "/scratch/$USER/$PBS_JOBID/remote/../turb/" -iname "*.bin" -exec ln -s {} "/scratch/$USER/$PBS_JOBID/16/turb/" \;

echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/"
echo 'current working directory:'
pwd
echo "START RUNNING JOBS IN find+xargs MODE"
WINEARCH="win32" WINEPREFIX="$HOME/.wine32" winefix
# run all the PBS *.p files in find+xargs mode
echo "following cases will be run from following path:"
echo "remote/pbs_in/dlc01_demos/"
export LAUNCH_PBS_MODE=false
/home/MET/sysalt/bin/find 'remote/pbs_in/dlc01_demos/' -type f -name '*.p' | sort -z

echo "number of files to be launched: "`find "remote/pbs_in/dlc01_demos/" -type f | wc -l`
/home/MET/sysalt/bin/find 'remote/pbs_in/dlc01_demos/' -type f -name '*.p' -print0 | sort -z | /home/MET/sysalt/bin/xargs -0 -I{} --process-slot-var=CPU_NR -n 1 -P 17 sh {}
echo "END OF JOBS IN find+xargs MODE"


echo "----------------------------------------------------------------------"
echo "total scratch disk usage:"
du -hs "/scratch/$USER/$PBS_JOBID/"
cd "/scratch/$USER/$PBS_JOBID/remote"
echo "current working directory:"
pwd
echo "Results saved at sim_id directory:"
find .

echo "move statsdel into compressed archive"
find "res/dlc01_demos/" -name "*.csv" -print0 | xargs -0 tar --remove-files -rf "prepost/statsdel_chnk_00000.tar"
xz -z2 -T 17 "prepost/statsdel_chnk_00000.tar"

echo "move log analysis into compressed archive"
find "logfiles/dlc01_demos/" -name "*.csv" -print0 | xargs -0 tar --remove-files -rf "prepost/loganalysis_chnk_00000.tar"
xz -z2 -T 17 "prepost/loganalysis_chnk_00000.tar"


echo "----------------------------------------------------------------------"
cd "/scratch/$USER/$PBS_JOBID/"
echo 'current working directory:'
pwd
echo "move results back from node scratch/sim_id to origin, but ignore htc, and pbs_in directories."
echo "copy from remote/ to $PBS_O_WORKDIR/"
time rsync -au "remote/" "$PBS_O_WORKDIR/" \
    --exclude "pbs_in/dlc01_demos/*" \
    --exclude *.htc
source deactivate
echo "DONE !!"

echo "----------------------------------------------------------------------"
# in case wine has crashed, kill any remaining wine servers
# caution: ALL the users wineservers will die on this node!
echo "following wineservers are still running:"
ps -u $USER -U $USER | grep wineserver
killall -u $USER wineserver
exit
