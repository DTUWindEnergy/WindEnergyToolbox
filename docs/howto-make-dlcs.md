Auto-generation of Design Load Cases
====================================


<!---
TODO, improvements:
putty reference and instructions (fill in username in the address username@gorm
how to mount gorm home on windows
do as on Arch Linux wiki: top line is the file name where you need to add stuff
point to the gorm/jess wiki's
explain the difference in the paths seen from a windows computer and the cluster
-->

> WARNING: these notes contain configuration settings that are specif to the
DTU Wind Energy cluster Gorm. Only follow this guide in another environment if
you know what you are doing!


Introduction
------------

For the auto generation of load cases and the corresponding execution on the
cluster, the following events will take place:
* Create an htc master file, and define the various tags in the exchange files
(spreadsheets).
* Generate the htc files for all the corresponding load cases based on the
master file and the tags defined in the exchange files. Besides the HAWC2 htc
input file, a corresponding pbs script is created that includes the instructions
to execute the relevant HAWC2 simulation on a cluster node. This includes copying
the model to the node scratch disc, executing HAWC2, copying the results from
the node scratch disc back to the network drive.
* Submit all the load cases (or the pbs launch scripts) to the cluster queueing
system. This is also referred to as launching the jobs.

Important note regarding file names. On Linux, file names and paths are case
sensitive, but on Windows they are not. Additionally, HAWC2 will always generate
result and log files with lower case file names, regardless of the user input.
Hence, in order to avoid possible ambiguities at all times, make sure that there
are no upper case symbols defined in the value of the following tags (as defined
in the Excel spreadsheets): ```[Case folder]```,  ```[Case id.]```, and
```[Turb base name]```.

The system will always force the values of the tags to be lower case anyway, and
when working on Windows, this might cause some confusing and unexpected behavior.
The tags themselves can have lower and upper case characters as can be seen
in the example above.

Notice that throughout the document ```$USER``` refers the your user name. You can
either let the system fill that in for you (by using the variable ```$USER```),
or explicitly user your user name instead. This user name is the same as your
DTU account name (or student account/number).

This document refers to commands to be entered in the terminal on Gorm when the
line starts with ```g-000 $```. The command that needs to be entered starts
after the ```$```.


Connecting to the cluster
-------------------------

You connect to the cluster via an SSH terminal. SSH is supported out of the box
for Linux and Mac OSX terminals (such as bash), but requires a separate
terminal client under Windows. Windows users are advised to use PuTTY and can
be downloaded at:
[http://www.chiark.greenend.org.uk/~sgtatham/putty/](http://www.chiark.greenend.org.uk/~sgtatham/putty/).
Here's a random
[tutorial](http://www.ghacks.net/2008/02/09/about-putty-and-tutorials-including-a-putty-tutorial/),
you can use your favourite search engine if you need more or different instructions.
More answers regarding PuTTY can also be found in the online
[documentation](http://the.earth.li/~sgtatham/putty/latest/htmldoc/).

The cluster that is setup for using the pre- and post-processing tools for HAWC2
has the following address: ```gorm.risoe.dk```.

On Linux/Mac connecting to the cluster is as simple as running the following
command in the terminal:

```
ssh $USER@gorm.risoe.dk
```

Use your DTU password when asked. This will give you terminal access to the
cluster called Gorm.

The cluster can only be reached when on the DTU network (wired, or only from a
DTU computer when using a wireless connection), when connected to the DTU VPN,
or from one of the DTU [databars](http://www.databar.dtu.dk/).

More information about the cluster can be found on the
[Gorm-wiki](http://gorm.risoe.dk/gormwiki)


Mounting the cluster discs
--------------------------

You need to be connected to the DTU network in order for this to work. You can
also connect to the DTU network over VPN. When doing the HAWC2 simulations, you
will interact regularly with the cluster file system and discs.

### Windows

It is convenient to map these discs as network drives (in Windows terms).
Map the following network drives (replace ```$USER``` with your user name):

```
\\mimer\hawc2sim
\\gorm\$USER # this is your Gorm home directory
```

Alternatively, on Windows you can use [WinSCP](http://winscp.net) to interact
with the cluster discs.

Note that by default Windows Explorer will hide some of the files you will need edit.
In order to show all files on your Gorm home drive, you need to un-hide system files:
Explorer > Organize > Folder and search options > select tab "view" > select the
option to show hidden files and folders.

### Unix

From Linux/Mac, you should be able to mount using either of the following
addresses:
```
//mimer.risoe.dk/hawc2sim
//gorm.risoe.dk/$USER
```
You can use either ```sshfs``` or ```mount -t cifs``` to mount the discs.


Preparation
-----------

Add the cluster-tools script to your system's PATH of you Gorm environment,
by editing the file ```.bash_profile``` file in your Gorm’s home directory
(```/home/$USER/.bash_profile```), and add the following lines (add at the end,
or create a new file with this file name in case it doesn't exist):

```
export PATH=$PATH:/home/MET/repositories/toolbox/pbsutils/
```

(The corresponding open repository is on the DTU Wind Energy Gitlab server:
[pbsutils](https://gitlab.windenergy.dtu.dk/toolbox/pbsutils). Please
considering reporting bugs and/or suggest improvements there. You're contributions
are much appreciated!)

> If you have been using an old version of this how-to, you might be pointing
to an earlier version of these tools/utils and any references containing
```cluster-tools``` or ```prepost``` should be removed
from your ```.bash_profile``` and/or ```.bashrc``` file on your gorm home drive.

After modifying ```.bash_profile```, save and close it. Then, in the terminal,
run the command (or logout and in again to be safe):
```
g-000 $ source ~/.bash_profile
```

You will also need to configure wine and place the HAWC2 executables in your
local wine directory, which by default is assumed to be ```~/.wine32```, and
```pbsutils``` contains and automatic configuration script you can run:

```
g-000 $ config-wine-hawc2.sh
```

If you need more information on what is going on, you can read a more detailed
description [here]
(https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/configure-wine.md).

All your HAWC2 executables and DLL's are now located
at ```/home/$USER/.wine32/drive_c/bin```.

Notice that the HAWC2 executable names are ```hawc2-latest.exe```,
```hawc2-118.exe```, etc. By default the latest version will be used and the user
does not need to specify this. However, when you need to compare different version
you can easily do so by specifying which case should be run with which
executable.

Alternatively you can also include all the DLL's and executables in the root of
your HAWC2 model folder. Executables and DLL's placed in the root folder take
precedence over the ones placed in ```/home/$USER/.wine32/drive_c/bin```.

Log out and in again from the cluster (close and restart PuTTY).

At this stage you can run HAWC2 as follows:

```
g-000 $ wine32 hawc2-latest htc/some-intput-file.htc
```


Updating local HAWC2 executables
--------------------------------

When there is a new version of HAWC2, or when a new license manager is released,
you can update your local wine directory as follows:

```
g-000 $ cp /home/MET/hawc2exe/* /home/$USER/.wine32/drive_c/bin/
```

The file ```hawc2-latest.exe``` will always be the latest HAWC2
version at ```/home/MET/hawc2exe/```. When a new HAWC2 is released you can
simply copy all the files from there again to update.


HAWC2 model folder structure and results on mimer/hawc2sim
----------------------------------------------------------

See [house rules on mimer/hawc2sim]
(https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/houserules-mimerhawc2sim.md)
for a more detailed description.


Method A: Generating htc input files on the cluster (recommended)
-----------------------------------------------------------------

Use ssh (Linux, Mac) or putty (MS Windows) to connect to the cluster.

In order to simplify things, we're using ```qsub-wrap.py``` from ```pbsutils```
(which we added under the [preparation]/(#preparation) section) in order to
generate the htc files. It will execute, on a compute node, any given Python
script in a pre-installed Python environment that has the Wind Energy Toolbox
installed.

For the current implementation of the DLB the following template is available:

```
/home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py
```

And the corresponding definitions of all the different load cases can be copied
from here (valid for the DTU10MW):

```
/mnt/mimer/hawc2sim/DTU10MW/C0020/htc/DLCs
```

For example, in order to generate all the HAWC2 htc input files and the
corresponding ```*.p``` cluster launch files using this default DLB setup with:

```
g-000 $ cd /mnt/mimer/hawc2sim/demo/A0001 # folder where the hawc2 model is located
g-000 $ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --prep
```

You could consider adding ```dlctemplate.py``` into the turbine folder or in
the simulation set id folder for your convenience:

```
g-000 $ cd /mnt/mimer/hawc2sim/demo/
# copy the dlctemplate to your turbine model folder and rename to myturbine.py
g-000 $ cp /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py ./myturbine.py
g-000 $ cd A0001
g-000 $ qsub-wrap.py -f ../myturbine.py --prep
```


Method B: Generating htc input files interactively on the cluster
-----------------------------------------------------------------

Use ssh (Linux, Mac) or putty (MS Windows) to connect to the cluster.

This approach gives you more flexibility, but requires more commands, and is hence
considered more difficult compared to method A.

First activate the Anaconda Python environment by typing:

```bash
# add the Anaconda Python environment paths to the system PATH
g-000 $ export PATH=/home/python/miniconda3/bin:$PATH
# activate the custom python environment:
g-000 $ source activate wetb_py3
```
For example, launch the auto-generation of DLCs input files:

```
# folder where the HAWC2 model is located
g-000 $ cd /mnt/mimer/hawc2sim/demo/AA0001
# assuming myturbine.py is copy of dlctemplate.py and is placed one level up
g-000 $ python ../myturbine.py --prep
```

Or start an interactive IPython shell:

```
g-000 $ ipython
```

Users should be aware that running computational heavy loads on the login node
is strictly discouraged. By overloading the login node other users will
experience slow login procedures, and the whole cluster could potentially be
jammed.


Method C: Generating htc input files locally
--------------------------------------------

This approach gives you more flexibility and room for custimizations, but you
will need to install a Python environment with all its dependencies locally.
Additionally, you need access to the cluster discs from your local workstation.

The installation procedure for wetb is outlined in the [installation manual]
(https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/install-manual-detailed.md).


Optional configuration
----------------------

Optional tags that can be set in the Excel spreadsheet and their corresponding
default values are given below. Beside a replacement value in the master htc
file, there are also special actions connected to these values. Consequently,
these tags have to be present. When removed, the system will stop working properly.

Relevant for the generation of the PBS launch scripts (```*.p``` files):
* ```[walltime] = '04:00:00' (format: HH:MM:SS)```
* ```[hawc2_exe] = 'hawc2-latest'```

Following directories have to be defined, and their default values are used when
they are not set explicitly in the spreadsheets.
* ```[animation_dir] = 'animation/'```
* ```[control_dir] = 'control/'```, all files and sub-folders copied to node
* ```[data_dir] = 'data/'```, all files and sub-folders copied to node
* ```[eigenfreq_dir] = False```
* ```[htc_dir] = 'htc/'```
* ```[log_dir] = 'logfiles/'```
* ```[res_dir] = 'res/'```
* ```[turb_dir] = 'turb/'```
* ```[turb_db_dir] = '../turb/'```
* ```[turb_base_name] = 'turb_'```

Required, and used for the PBS output and post-processing
* ```[pbs_out_dir] = 'pbs_out/'```
* ```[iter_dir] = 'iter/'```

Optional
* ```[turb_db_dir] = '../turb/'```
* ```[wake_dir] = False```
* ```[wake_db_dir] = False```
* ```[wake_base_name] = 'turb_'```
* ```[meander_dir] = False```
* ```[meand_db_dir] = False```
* ```[meand_base_name] = 'turb_'```
* ```[mooring_dir] = False```, all files and sub-folders copied to node
* ```[hydro_dir] = False```, all files and sub-folders copied to node

A zip file will be created which contains all files in the model root directory,
and all the contents (files and folders) of the following directories:
```[control_dir], [mooring_dir], [hydro_dir], 'externalforce/', [data_dir]```.
This zip file will be extracted into the execution directory (```[run_dir]```).
After the model has ran on the node, only the files that have been created
during simulation time in the ```[log_dir]```, ```[res_dir]```,
```[animation_dir]```, and ```[eigenfreq_dir]``` will be copied back.


### Advanced configuration options

> Note that not all features are documented yet...

Special tags: copy special result files from the compute node back to the HAWC2
working directory on the network drive, and optionally rename the file in case
it would otherwise be overwritten by other cases in your DLB:
* ```[copyback_files] = ['ESYSMooring_init.dat']```
* ```[copyback_frename] = ['path/to/ESYSMooring_init_vXYZ.dat']```, optionally specify
a different file path/name

Copy files from the HAWC2 working directory with a special name to the compute
node for which the a fixed file name is assumed
* ```[copyto_files] = ['path/to/ESYSMooring_init_vXYZ.dat']```
* ```[copyto_generic] = ['ESYSMooring_init.dat']```


Launching the jobs on the cluster
---------------------------------

Use ssh (Linux, Mac) or putty (MS Windows) to connect to the cluster.

The ```launch.py``` is a generic tool that helps with launching an arbitrary
number of pbs launch script on a PBS Torque cluster. Launch scripts here
are defined as files with a ```.p``` extension. The script will look for any
```.p``` files in a specified folder (```pbs_in/``` by default, which the user
can change using the  ```-p``` or ```--path_pbs``` flag) and save them in a
file list called ```pbs_in_file_cache.txt```. When using the option ```-c``` or
```--cache```, the script will not look for pbs files, but instead read them
directly from the ```pbs_in_file_cache.txt``` file.

The launch script has a simple build in scheduler that has been successfully
used to launch 50.000 jobs. This scheduler is configured by two parameters:
number of cpu's requested (using ```-c``` or ```--nr_cpus```) and minimum
of required free cpu's on the cluster (using ```--cpu_free```, 48 by default).
Jobs will be launched after a predefined sleep time (as set by the
```--tsleep``` option, and set to 5 seconds by default). After the initial sleep
time a new job will be launched every 0.1 second. If the launch condition is not
met (```nr_cpus > cpu's used by user AND cpu's free on cluster > cpu_free```),
the program will wait 5 seconds before trying to launch a new job again.

Depending on the amount of jobs and the required computation time, it could
take a while before all jobs are launched. When running the launch script from
the login node, this might be a problem when you have to close your ssh/putty
session before all jobs are launched. In that case the user should use a
dedicated compute node for launching jobs. To run the launch script on a
compute instead of the login node, use the ```--node``` option. You can inspect
the progress in the ```launch_scheduler_log.txt``` file.

The ```launch.py``` script has some different options, and you can read about
them by using the help function (the output is included for your convenience):

```bash
g-000 $ launch.py --help
usage:

launch.py -n nr_cpus

launch.py --crontab when running a single iteration of launch.py as a crontab job every 5 minutes.
File list is read from "launch_pbs_filelist.txt", and the configuration can be changed on the fly
by editing the file "launch_scheduler_config.txt".

options:
  -h, --help            show this help message and exit
  --depend              Switch on for launch depend method
  -n NR_CPUS, --nr_cpus=NR_CPUS
                        number of cpus to be used
  -p PATH_PBS_FILES, --path_pbs_files=PATH_PBS_FILES
                        optionally specify location of pbs files
  --re=SEARCH_CRIT_RE   regular expression search criterium applied on the
                        full pbs file path. Escape backslashes! By default it
                        will select all *.p files in pbs_in/.
  --dry                 dry run: do not alter pbs files, do not launch
  --tsleep=TSLEEP       Sleep time [s] after qsub command. Default=5 seconds
  --logfile=LOGFILE     Save output to file.
  -c, --cache           If on, files are read from cache
  --cpu_free=CPU_FREE   No more jobs will be launched when the cluster does
                        not have the specified amount of cpus free. This will
                        make sure there is room for others on the cluster, but
                        might mean less cpus available for you. Default=48
  --cpu_user_queue=CPU_USER_QUEUE
                        No more jobs will be launched after having
                        cpu_user_queue number of jobs in the queue. This
                        prevents users from filling the queue, while still
                        allowing to aim for a high cpu_free target.
                        Default=500
  --qsub_cmd=QSUB_CMD   Is set automatically by --node flag
  --node                If executed on dedicated node. Although this works,
                        consider using --crontab instead. Default=False
  --sort                Sort pbs file list. Default=False
  --crontab             Crontab mode. Implies --cache, and not compatible with
                        --node. Default=False
  --debug               Debug print statements. Default=False

```

Then launch the actual jobs (each job is a ```*.p``` file in ```pbs_in```) using
100 cpu's:

```bash
g-000 $ cd /mnt/mimer/hawc2sim/demo/A0001
g-000 $ launch.py -n 100 -p pbs_in/
```

If the launching process requires hours, and you have to close you SHH/PuTTY
session before it reaches the end, you can either use the ```--node``` or the
```--crontab``` argument. When using ```--node```, ```launch.py``` will run on
a dedicated cluster note, submitted as a PBS job. When using ```--crontab```,
```launch.py``` will be run once every 5 minutes as a ```crontab``` job on the
login node. This is preferred since you are not occupying a node with a very
simple and light job.

```bash
g-000 $ cd /mnt/mimer/hawc2sim/demo/A0001
g-000 $ launch.py -n 100 -p pbs_in/ --crontab
```


Inspecting running jobs
-----------------------

There are a few tools you can use from the command line to see what is going on
the cluster. How many nodes are free, how many nodes do I use as a user, etc.

* ```cluster-status.py``` overview dashboard of the cluster: nodes free, running,
length of the queue, etc
* ```qstat -u $USER``` list all the running and queued jobs of the user
* ```nnsqdel $USER all``` delete all the jobs that from the user
* ```qdel_range JOBID_FROM JOBID_TIL``` delete a range of job id's

Notice that the pbs output files in ```pbs_out``` are only created when the job
has ended (or failed). When you want to inspect a running job, you can ssh from
the Gorm login node to node that runs the job. First, find the job id by listing
all your current jobs (```qstat -u $USER```). The job id can be found in the
first column, and you only need to consider the number, not the domain name
attached to it. Now find the on which node it runs with (replace 123546 with the
relevant job id):
```
g-000 $ qstat -f 123456 | grep exec_host
```

From here you login into the node as follows (replace g-078 with the relevant
node):
```
g-000 $ ssh g-078
```

And browse to the scratch directory which lands you in the root directory of
your running HAWC2 model (replace 123456 with the relevant job id):
```
g-000 $ cd /scratch/$USER/123456.g-000.risoe.dk
```

You can find what HAWC2 (or whatever other executable you are running) is
outputting to the command line in the file:
```
/var/lib/torque/spool/JOBID.jess.dtu.dk.OU
```
Or when watch what is happening at the end in real time
```
tail -f /var/lib/torque/spool/JOBID.jess.dtu.dk.OU
```


Re-launching failed jobs
------------------------

In case you want to re-launch only a subset of a previously generated set of
load cases, there are several methods:

1. Copy the PBS launch scripts (they have the ```*.p``` extension and can be
found in the ```pbs_in``` folder) of the failed cases to a new folder (for
example ```pbs_in_failed```). Now run ```launch.py``` again, but instead point
to the folder that contains the ```*.p``` files of the failed cases, for example:
```
g-000 $ launch.py -n 100 --node -p pbs_in_failed
```

2. Use the ```--cache``` option, and edit the PBS file list in the file
```launch_pbs_filelist.txt``` so that only the simulations remain that have to be
run again. ```launch_pbs_filelist.txt``` is created every time you run
```launch.py```. You can use the option ```--dry``` to make a practice launch
run, and that will create a ```launch_pbs_filelist.txt``` file, but not a single
job will be launched.

3. Each pbs file can be launched manually as follows:
```
g-000 $ qsub path/to/pbs_file.p
```

Alternatively, one can use the following options in ```launch.py```:

* ```-p some/other/folder```: specify from which folder the pbs files should be taken
* ```--re=SEARCH_CRIT_RE```: advanced filtering based on the pbs file names. It
requires some notion of regular expressions (some random tutorials:
[1](http://www.codeproject.com/Articles/9099/The-Minute-Regex-Tutorial),
[2](http://regexone.com/))
    * ```launch.py -n 10 --re=.SOMENAME.``` will launch all pbs file that
    contains ```SOMENAME```. Notice the leading and trailing colon, which is
    in bash environments is equivalent to the wild card (*).


Post-processing
---------------

The post-processing happens through the same script as used for generating the
htc files, but now we set different flags. For example, for checking the log
files, calculating the statistics, the AEP and the life time equivalent loads:

```
# myturbine.py (copy of dlctemplate.py) is assumed to be located one folder up
g-000 $ qsub-wrap.py -f ../myturbine.py --years=25 --neq=1e7 --stats --check_logs --fatigue
```

Other options for the original ```dlctemplate.py``` script:

```
usage: dlctemplate.py [-h] [--prep] [--check_logs] [--stats] [--fatigue]
                      [--csv] [--years YEARS] [--no_bins NO_BINS] [--neq NEQ]
                      [--envelopeblade] [--envelopeturbine]

pre- or post-processes DLC's

optional arguments:
  -h, --help         show this help message and exit
  --prep             create htc, pbs, files (default=False)
  --check_logs       check the log files (default=False)
  --stats            calculate statistics and 1Hz equivalent loads (default=False)
  --fatigue          calculate Leq for a full DLC (default=False)
  --csv              Save data also as csv file (default=False)
  --years YEARS      Total life time in years (default=20)
  --no_bins NO_BINS  Number of bins for fatigue loads (default=46)
  --neq NEQ          Equivalent cycles neq, default 1 Hz equivalent load
                     (neq = simulation duration in seconds)
  --envelopeblade    calculate the load envelope for sensors on the blades
  --envelopeturbine  calculate the load envelope for sensors on the turbine
```

The load envelopes are computed for sensors specified in the
```myturbine.py``` file. The sensors are specified in a list of lists. The
inner list contains the sensors at one location. The envelope is computed for
the first two sensors of the inner list and the other sensors are used to
retrieve the remaining loads defining the load state occurring at the same
instant. The outer list is used to specify sensors at different locations.
The default values for the blade envelopes are used to compute the Mx-My
envelopes and retrieve the Mz-Fx-Fy-Fz loads occurring at the same moment.


Debugging
---------

Any output (everything that involves print statements) generated during the
post-processing of the simulations using ```myturbine.py``` is captured in
the ```pbs_out/qsub-wrap_myturbine.py.out``` file, while exceptions and errors
are redirected to the ```pbs_out/qsub-wrap_myturbine.py.err``` text file.

The output and errors of HAWC2 simulations can also be found in the ```pbs_out```
directory. The ```.err``` and ```.out``` files will be named exactly the same
as the ```.htc``` input files, and the ```.sel```/```.dat``` output files.

