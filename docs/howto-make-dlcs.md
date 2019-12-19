Auto-generation of Design Load Cases
====================================


<!---
TODO, improvements:
do as on Arch Linux wiki: top line is the file name where you need to add stuff
explain the difference in the paths seen from a windows computer and the cluster

DONE:
- putty reference and instructions (fill in username in the address
  username@gorm) [rink]
- how to mount gorm home on windows [rink]
- point to the gorm/jess wiki's [rink]
-->

> WARNING: these notes contain configuration settings that are specif to the
DTU Wind Energy cluster Jess. Only follow this guide in another environment if
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
```[turb_base_name]```.

The system will always force the values of the tags to be lower case anyway, and
when working on Windows, this might cause some confusing and unexpected behavior.
The tags themselves can have lower and upper case characters as can be seen
in the example above.

Notice that throughout the document ```$USER``` refers the your user name. You can
either let the system fill that in for you (by using the variable ```$USER```),
or explicitly user your user name instead. This user name is the same as your
DTU account name (or student account/number).

This document refers to commands to be entered in the terminal on Jess when the
line starts with ```j-000 $```. The command that needs to be entered starts
after the ```$```.


Pdap
----

You can also use the Pdap for post-processing, which includes a MS Word report
generator based on a full DLB, a GUI for easy plotting of HAWC2 result files,
and a Python scripting interface:

* [Pdap](http://www.hawc2.dk/Download/Post-processing-tools/Pdap)
* [Pdap report/docs](http://orbit.dtu.dk/en/publications/post-processing-of-design-load-cases-using-pdap%28827c432b-cf7d-44eb-899b-93e9c0648ca5%29.html)


Connecting to the cluster
-------------------------

We provide here an overview of how to connect to the cluster, but general,
up-to-date information can be found in the [HPC documentation](https://docs.hpc.ait.dtu.dk).

You connect to the cluster via an SSH terminal, and there are different SSH
terminals based on your operating system (see the platform-specific
instructions in the next subsections). The cluster can only be reached when
on the DTU network (wired, or only from a DTU computer when using a wireless
connection), when connected to the DTU VPN, or from one of the DTU
[databars](http://www.databar.dtu.dk/).

### Windows

Windows users are advised to use PuTTY, which can
be downloaded from
[this link](http://www.chiark.greenend.org.uk/~sgtatham/putty/).

Once you have installed PuTTY and placed the executable somewhere convenient
(e.g., the Desktop), double click on the executable. In the window that opens
up, enter/verify the following settings:
* Session > Host Name: jess.dtu.dk
* Session > Port: 22
* Session > Connection type: SSH
* Session > Saved Sessions: Jess
* Connection > Data > Auto-login username: your DTU username
* Connection > Data > When username is not specified: Use system username
* Window > Colours > Select a colour to adjust > ANSI Blue: RGB = 85, 85, 255
* Window > Colours > Select a colour to adjust > ANSI Bold Blue: RGB = 128, 128, 255

Note that these last two options are optional. We've found that the default
color for comments, ANSI Blue, is too dark to be seen on the black
background. The last two options in the list set ANSI Blue and ANSI Blue Bold
to be lighter and therefore easier to read when working in the terminal. Once
you have entered these options, click "Save" on the "Session" tab and close
the window.

With PuTTY configured, you can connect to Jess by double-clicking the PuTTY
executable; then, in the window that opens select "Jess" in "Saved Sessions",
click the "Load" button, and finally click the "Open" button. A terminal
window will open up. Type your DTU password in this new window when prompted
(your text will not appear in the window) and then hit the Enter key. You
should now be logged into Jess.

To close the PuTTY window, you can either hit the red "X" in the upper-right
corner of the window or type "exit" in the terminal and hit enter.

More information on using PuTTY and how it works can be found in this
[PuTTY tutorial](http://www.ghacks.net/2008/02/09/about-putty-and-tutorials-including-a-putty-tutorial/)
or in the online
[documentation](http://the.earth.li/~sgtatham/putty/latest/htmldoc/).
You are also welcome to use Google and read the many online resources.

### Unix

Unlike Windows, SSH is supported out of the box for Linux and Mac OSX
terminals. To connect to the cluster, enter the following command into
the terminal:

```
ssh $USER@jess.dtu.dk
```

Enter your DTU password when prompted. This will give you terminal access
to the Jess cluster.


Mounting the cluster discs
--------------------------

When doing the HAWC2 simulations, you will interact regularly with the cluster
file system and discs. Thus, it can be very useful to have two discs mounted
locally so you can easily access them: 1) your home directory on Jess and 2)
the HAWC2 simulation folder on Mimer.

You need to be connected to the DTU network (either directly or via VPN) for
the following instructions to work.


### Windows

On Windows, we recommend mapping the two drives to local network drives, which
means that you can navigate/copy/paste to/from them in Windows Explorer just as
you would do with normal folders on your computer. You may also use [WinSCP](http://winscp.net)
to interact with the cluster discs if you are more familiar with that option.

Here we provide instructions for mapping network drives in Windows 7. If these
instructions don't work for you, you can always find directions for your
version of Windows by Googling "map network drive windows $WIN_VERSION", where
$WIN_VERSION is your version number.

In Windows 7, you can map a network drive in the following steps:
1. Open a Windows Explorer window
2. Right-click on "Computer" and select "Map network drive"
3. Select any unused drive and type ```\\Jess.dtu.dk\$USER``` into the folder field,
replacing "$USER" with your DTU username (e.g., DTU user "ABCD" has a Jess home
drive of ```\\Jess.dtu.dk\abcd```)
4. Check the "Reconnect at logon" box if you want to connect to this drive
every time you log into your computer (recommended)
5. Click the Finish button
6. Repeat Steps 1 through 5, replacing the Jess home address in Step 3 with the
HAWC2 simulation folder address: ```\\mimer.risoe.dk\hawc2sim```

Note that by default Windows Explorer will hide some of the files you will need
edit. In order to show all files on your Jess home drive, you need to un-hide
system files: Explorer > Organize > Folder and search options > "View" tab >
Hidden files and folders > "Show hidden files, folders, and drives".

### Unix

From Linux/Mac, you should be able to mount using either of the following
addresses:
```
//mimer.risoe.dk/hawc2sim
//jess.dtu.dk/$USER
```
You can use either ```sshfs``` or ```mount -t cifs``` to mount the discs.


Preparation
-----------

Add the cluster-tools script to your system's PATH of you Jess environment,
by editing the file ```.bash_profile``` file in your Jess’s home directory
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
from your ```.bash_profile``` and/or ```.bashrc``` file on your Jess home drive.

After modifying ```.bash_profile```, save and close it. Then, in the terminal,
run the command (or logout and in again to be safe):
```
j-000 $ source ~/.bash_profile
j-000 $ source ~/.bashrc
```

You will also need to configure wine and place the HAWC2 executables in your
local wine directory, which by default is assumed to be ```~/.wine32```, and
```pbsutils``` contains and automatic configuration script you can run:

```
j-000 $ /home/MET/repositories/toolbox/pbsutils/config-wine-hawc2.sh
```

If you need more information on what is going on, you can read a more detailed
description [here]
(https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/configure-wine.md).

All your HAWC2 executables and DLL's are now located
at ```/home/$USER/wine_exe/win32```.

Notice that the HAWC2 executable names are ```hawc2-latest.exe```,
```hawc2-118.exe```, etc. By default the latest version will be used and the user
does not need to specify this. However, when you need to compare different version
you can easily do so by specifying which case should be run with which
executable.

Alternatively you can also include all the DLL's and executables in the root of
your HAWC2 model folder. Executables and DLL's placed in the root folder take
precedence over the ones placed in ```/home/$USER/wine_exe/win32```.

> IMPORTANT: log out and in again from the cluster (close and restart PuTTY)
> before trying to see if you can run HAWC2.

At this stage you can run HAWC2 as follows:

```
j-000 $ wine32 hawc2-latest htc/some-intput-file.htc
```


Updating local HAWC2 executables
--------------------------------

When there is a new version of HAWC2, or when a new license manager is released,
you can update your local wine directory as follows:

```
j-000 $ rsync -au /home/MET/hawc2exe/win32 /home/$USER/wine_exe/win32 --progress
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

Note that ```dlctemplate.py``` does not require any changes or modifications
if you are only interested in running the standard DLB as explained here.

For example, in order to generate all the HAWC2 htc input files and the
corresponding ```*.p``` cluster launch files using this default DLB setup with:

```
j-000 $ cd /mnt/mimer/hawc2sim/demo/A0001 # folder where the hawc2 model is located
j-000 $ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --prep
```

You could consider adding ```dlctemplate.py``` into the turbine folder or in
the simulation set id folder for your convenience:

```
j-000 $ cd /mnt/mimer/hawc2sim/demo/
# copy the dlctemplate to your turbine model folder and rename to myturbine.py
j-000 $ cp /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py ./myturbine.py
j-000 $ cd A0001
j-000 $ qsub-wrap.py -f ../myturbine.py --prep
```


Method B: Generating htc input files interactively on the cluster
-----------------------------------------------------------------

Use ssh (Linux, Mac) or putty (MS Windows) to connect to the cluster.

This approach gives you more flexibility, but requires more commands, and is hence
considered more difficult compared to method A.

First activate the Anaconda Python environment by typing:

```bash
# add the Anaconda Python environment paths to the system PATH
j-000 $ export PATH=/home/python/miniconda3/bin:$PATH
# activate the custom python environment:
j-000 $ source activate py36-wetb
```
For example, launch the auto-generation of DLCs input files:

```
# folder where the HAWC2 model is located
j-000 $ cd /mnt/mimer/hawc2sim/demo/AA0001
# assuming myturbine.py is copy of dlctemplate.py and is placed one level up
j-000 $ python ../myturbine.py --prep
```

Or start an interactive IPython shell:

```
j-000 $ ipython
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

The installation procedure for wetb is outlined in the
[simple user](docs/install.md) or the
[developer/contributor](docs/developer-guide.md) installation manual.


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
* ```[micro_dir] = False```
* ```[micro_db_dir] = False```
* ```[micro_base_name] = 'turb_'```
* ```[meander_dir] = False```
* ```[meander_db_dir] = False```
* ```[meander_base_name] = 'turb_'```
* ```[mooring_dir] = False```, all files and sub-folders copied to node
* ```[hydro_dir] = False```, all files and sub-folders copied to node

The mooring line dll has a fixed name init file that has to be in the root of
the HAWC2 folder. When you have to use various init files (e.g. when the water
depth is varying for different load cases) it would be convienent to be able
to control which init file is used for which case (e.g. water depth).

When running a load case for which the mooring lines will run in init mode:
* ```[copyback_f1] = 'ESYSMooring_init.dat'```
* ```[copyback_f1_rename] = 'mooringinits/ESYSMooring_init_vXYZ.dat'```

When using an a priory cacluated init file for the mooring lines:
* ```[copyto_f1] = 'mooringinits/ESYSMooring_init_vXYZ.dat'```
* ```[copyto_generic_f1] = 'ESYSMooring_init.dat'```

Replace ```vXYZ``` with an appropriate identifier for your case.

A zip file will be created which contains all files in the model root directory,
and all the contents (files and folders) of the following directories:
```[control_dir], [mooring_dir], [hydro_dir], 'externalforce/', [data_dir]```.
This zip file will be extracted into the execution directory (```[run_dir]```).
After the model has ran on the node, only the files that have been created
during simulation time in the ```[log_dir]```, ```[res_dir]```,
```[animation_dir]```, and ```[eigenfreq_dir]``` will be copied back.


### Advanced configuration options by modifying dlctemplate.py

> Note that not all features are documented yet...

Special tags: copy special result files from the compute node back to the HAWC2
working directory on the network drive, and optionally rename the file in case
it would otherwise be overwritten by other cases in your DLB:
* ```[copyback_files] = ['ESYSMooring_init.dat']```
* ```[copyback_frename] = ['path/to/ESYSMooring_init_vXYZ.dat']```, optionally
specify a different file path/name

Copy files from the HAWC2 working directory with a special name to the compute
node for which the a fixed file name is assumed
* ```[copyto_files] = ['path/to/ESYSMooring_init_vXYZ.dat']```
* ```[copyto_generic] = ['ESYSMooring_init.dat']```


### Tags required for standalone Mann 64-bit turbulence generator

```dlctemplate.py``` has a flag named ```--pbs_turb```, which when activated
generates PBS input files containing the instructions to generate all required
turbulence boxes using the 64-bit version of the stand alone Mann turbulence
box generator. The appropriate input parameters are taken from the following
tags:

* Atmospheric turbulence:
    * ```[tu_model] = 1```
    * ```[turb_base_name]```
    * ```[MannAlfaEpsilon]```
    * ```[MannL]```
    * ```[MannGamma]```
    * ```[seed]```
    * ```[turb_nr_u]``` : number of grid points in the u direction
    * ```[turb_nr_v]``` : number of grid points in the v direction
    * ```[turb_nr_w]``` : number of grid points in the w direction
    * ```[turb_dx]``` : grid spacing in meters in the u direction
    * ```[turb_dy]``` : grid spacing in meters in the v direction
    * ```[turb_dz]``` : grid spacing in meters in the w direction
    * ```[high_freq_comp]```

* Micro turbulence for DWM:
    * ```[micro_base_name]```
    * ```[MannAlfaEpsilon_micro]```
    * ```[MannL_micro]```
    * ```[MannGamma_micro]```
    * ```[seed_micro]```
    * ```[turb_nr_u_micro]``` : number of grid points in the u direction
    * ```[turb_nr_v_micro]``` : number of grid points in the v direction
    * ```[turb_nr_w_micro]``` : number of grid points in the w direction
    * ```[turb_dx_micro]``` : grid spacing in meters in the u direction
    * ```[turb_dy_micro]``` : grid spacing in meters in the v direction
    * ```[turb_dz_micro]``` : grid spacing in meters in the w direction
    * ```[high_freq_comp_micro]```

* Meander turbulence for DWM
    * ```[meander_base_name]```
    * ```[MannAlfaEpsilon_meander]```
    * ```[MannL_meander]```
    * ```[MannGamma_meander]```
    * ```[seed_meander]```
    * ```[turb_nr_u_meander]``` : number of grid points in the u direction
    * ```[turb_nr_v_meander]``` : number of grid points in the v direction
    * ```[turb_nr_w_meander]``` : number of grid points in the w direction
    * ```[turb_dx_meander]``` : grid spacing in meters in the u direction
    * ```[turb_dy_meander]``` : grid spacing in meters in the v direction
    * ```[turb_dz_meander]``` : grid spacing in meters in the w direction
    * ```[high_freq_comp_meander]```


### Tags required for hydro file generation

* ```[hydro_dir]```
* ```[hydro input name]```
* ```[wave_type]``` : see HAWC2 manual for options
* ```[wave_spectrum]``` : see HAWC2 manual for options
* ```[wdepth]```
* ```[Hs]``` : see HAWC2 manual for options
* ```[Tp]``` : see HAWC2 manual for options
* ```[wave_seed]``` : see HAWC2 manual for options
* ```[wave_gamma]``` : see HAWC2 manual for options
* ```[wave_coef]``` : see HAWC2 manual for options
* ```[stretching]``` : see HAWC2 manual for options
* ```[embed_sf]``` : see HAWC2 manual for options, and look for how it is implemented
in [prepost.dlcsdefs.vartag_excel_stabcon(master)](wetb/prepost/dlcdefs.py).

And the corresponding section the htc master file:

```
begin hydro;
  begin water_properties;
    rho 1027 ; kg/m^3
    gravity 9.81 ; m/s^2
    mwl 0.0;
    mudlevel [wdepth];
    wave_direction [wave_dir];
    water_kinematics_dll ./wkin_dll.dll   ./[hydro_dir][hydro input name].inp;
  end water_properties;
end hydro;
```


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
time a new job will be launched every 0.5 second. If the launch condition is not
met:

```
nr_cpus > cpu's used by user
AND cpu's free on cluster > cpu_free
AND jobs queued by user < cpu_user_queue
```

the program will sleep 5 seconds before trying to launch a new job again.

Depending on the amount of jobs and the required computation time, it could
take a while before all jobs are launched. When running the launch script from
the login node, this might be a problem when you have to close your ssh/putty
session before all jobs are launched. In that case the user can use the
```--crontab``` argument: it will trigger the ```launch.py``` script every 5
minutes to check if more jobs can be launched until all jobs have been
executed. The user does not need to have an active ssh/putty session for this to
work. You can follow the progress and configuration of ```launch.py``` in
crontab mode in the following files:

* ```launch_scheduler_log.txt```
* ```launch_scheduler_config.txt```: you can change your launch settings on the fly
* ```launch_scheduler_state.txt```
* ```launch_pbs_filelist.txt```: remaining jobs, when a job is launched it is
removed from this list

You can check if ```launch.py``` is actually active as a crontab job with:

```
crontab -l
```

```launch.py``` will clean-up the crontab after all jobs are launched, but if
you need to prevent it from launching new jobs before that, you can clean up your
crontab with:

```
crontab -r
```

The ```launch.py``` script has various different options, and you can read about
them by using the help function (the output is included for your convenience):

```bash
j-000 $ launch.py --help
Usage:

launch.py -n nr_cpus

launch.py --crontab when running a single iteration of launch.py as a crontab job every 5 minutes.
File list is read from "launch_pbs_filelist.txt", and the configuration can be changed on the fly
by editing the file "launch_scheduler_config.txt".

Options:
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
  --tsleep=TSLEEP       Sleep time [s] when cluster is too bussy to launch new
                        jobs. Default=5 seconds
  --tsleep_short=TSLEEP_SHORT
                        Sleep time [s] between between successive job
                        launches. Default=0.5 seconds.
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
                        allowing to aim for a high cpu_free target. Default=5
  --qsub_cmd=QSUB_CMD   Is set automatically by --node flag
  --node                If executed on dedicated node. Although this works,
                        consider using --crontab instead. Default=False
  --sort                Sort pbs file list. Default=False
  --crontab             Crontab mode: %prog will check every 5 (default)
                        minutes if more jobs can be launched. Not compatible
                        with --node. When all jobs are done, crontab -r will
                        remove all existing crontab jobs of the current user.
                        Use crontab -l to inspect current crontab jobs, and
                        edit them with crontab -e. Default=False
  --every_min=EVERY_MIN
                        Crontab update interval in minutes. Default=5
  --debug               Debug print statements. Default=False

```

Then launch the actual jobs (each job is a ```*.p``` file in ```pbs_in```) using
100 cpu's:

```bash
j-000 $ cd /mnt/mimer/hawc2sim/demo/A0001
j-000 $ launch.py -n 100 -p pbs_in/
```

If the launching process requires hours, and you have to close you SHH/PuTTY
session before it reaches the end, you can either use the ```--node``` or the
```--crontab``` argument. When using ```--node```, ```launch.py``` will run on
a dedicated cluster node, submitted as a PBS job. When using ```--crontab```,
```launch.py``` will be run once every 5 minutes as a ```crontab``` job on the
login node. This is preferred since you are not occupying a node with a very
simple and light job. ```launch.py``` will remove all the users crontab jobs
at the end with ```crontab -r```.

```bash
j-000 $ cd /mnt/mimer/hawc2sim/demo/A0001
j-000 $ launch.py -n 100 -p pbs_in/ --crontab
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
the Jess login node to node that runs the job. First, find the job id by listing
all your current jobs (```qstat -u $USER```). The job id can be found in the
first column, and you only need to consider the number, not the domain name
attached to it. Now find the on which node it runs with (replace 123546 with the
relevant job id):
```
j-000 $ qstat -f 123456 | grep exec_host
```

From here you login into the node as follows (replace j-078 with the relevant
node):
```
j-000 $ ssh j-078
```

And browse to the scratch directory which lands you in the root directory of
your running HAWC2 model (replace 123456 with the relevant job id):
```
j-000 $ cd /scratch/$USER/123456.j-000.risoe.dk
```

You can find what HAWC2 (or whatever other executable you are running) is
outputting to the command line in the file:
```
/var/lib/torque/spool/JOBID.jess.dtu.dk.OU
```
Or when watch what is happening at the end in real time
```
# on Jess:
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
j-000 $ launch.py -n 100 --node -p pbs_in_failed
```

2. Use the ```--cache``` option, and edit the PBS file list in the file
```launch_pbs_filelist.txt``` so that only the simulations remain that have to be
run again. ```launch_pbs_filelist.txt``` is created every time you run
```launch.py```. You can use the option ```--dry``` to make a practice launch
run, and that will create a ```launch_pbs_filelist.txt``` file, but not a single
job will be launched.

3. Each pbs file can be launched manually as follows:
```
j-000 $ qsub path/to/pbs_file.p
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
j-000 $ qsub-wrap.py -f ../myturbine.py --years=25 --neq=1e7 --stats --check_logs --fatigue
```

Other options for the original ```dlctemplate.py``` script:

```
(py36-wetb) [dave@jess]$ python dlctemplate.py --help
usage: dlctemplate.py [-h] [--prep] [--check_logs]
                      [--pbs_failed_path PBS_FAILED_PATH] [--stats]
                      [--fatigue] [--AEP] [--csv] [--years YEARS]
                      [--no_bins NO_BINS] [--neq NEQ] [--rotarea ROTAREA]
                      [--save_new_sigs] [--dlcplot] [--envelopeblade]
                      [--envelopeturbine] [--zipchunks] [--pbs_turb]
                      [--walltime WALLTIME]

pre- or post-processes DLC's

optional arguments:
  -h, --help            show this help message and exit
  --prep                create htc, pbs, files
  --check_logs          check the log files
  --pbs_failed_path PBS_FAILED_PATH
                        Copy pbs launch files of the failed cases to a new
                        directory in order to prepare a re-run. Default value:
                        pbs_in_failed.
  --stats               calculate statistics and 1Hz equivalent loads
  --fatigue             calculate Leq for a full DLC
  --AEP                 calculate AEP, requires htc/DLCs/dlc_config.xlsx
  --csv                 Save data also as csv file
  --years YEARS         Total life time in years
  --no_bins NO_BINS     Number of bins for fatigue loads
  --neq NEQ             Equivalent cycles Neq used for Leq fatigue lifetime
                        calculations.
  --rotarea ROTAREA     Rotor area for C_T, C_P
  --save_new_sigs       Save post-processed sigs
  --dlcplot             Plot DLC load basis results
  --envelopeblade       Compute envelopeblade
  --envelopeturbine     Compute envelopeturbine
  --zipchunks           Create PBS launch files forrunning in zip-chunk
                        find+xargs mode.
  --pbs_turb            Create PBS launch files to create the turbulence boxes
                        in stand alone mode using the 64-bit Mann turbulence
                        box generator. This can be usefull if your turbulence
                        boxes are too big for running in HAWC2 32-bit mode.
                        Only works on Jess.
  --walltime WALLTIME   Queue walltime for each case/pbs file, format:
                        HH:MM:SS Default: 04:00:00

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

