# Getting started with generating DLBs for HAWC2

Note that DLB stands for Design Load Basis. It refers to a set of cases that are
used to evaluate the fitness of a certain design. An example of a DLB definition
is the IEC 61400-1ed3.


## Overview

This document intends to provide an extremely brief overview of how to run a set
of HAWC2 simulations using the Gorm cluster at DTU and the Mimer storage.
This document is a work in progress, and is by no means exhaustive.


## Resources

The majority of this information can be found in the Wind Energy Toolbox
documentation. In particular, [generate-spreadsheet](docs/generate-spreadsheet.md)
discusses how to use a "master" Excel spreadsheet to generate the subordinate
Excel spreadsheets that will later be used to create the necessary HTC files.
[howto-make-dlcs](docs/howto-make-dlcs.md) discusses how to create htc files
from the subordinate spreadsheets, submit those HTC files to the cluster,
and post-process results.
[houserules-mimerhawc2sim](docs/houserules-mimerhawc2sim.md) has some
"house rules" on storing simulations on mimer.
[using-statistics-df.md](docs/using-statistics-df) has some information
on loading the post-processing statistics using Python.


## Steps

##### 1. Make sure that you can access the cluster/mimer.
See the instructions on [this page](docs/howto-make-dlcs.md).

##### 2. Create a Set ID folder for this project/simulation.
You should find that, within a given turbine model, the folder structure is
similar to the following:

```
|-- DTU10MW/
|   |-- AA0001
|   |   |-- ...
|   |-- AA0002
|   |   |-- ...
|   |-- ...
|   |-- AB0001
|   |-- ...
|-- AA_log_DTUMW.xlsx
|-- AB_log_DTUMW.xlsx
|-- ...
```

Here, each of these alphanumeric folders are "set IDs", and you should have a
unique set ID for each set of simulations. Detailed house rules on how you
should store data on mimer can be found in the
[houserules-mimerhawc2sim](docs/houserules-mimerhawc2sim.md) document.

There are two steps to creating your new set ID folder:
1. Determine if you need to create a new turbine model folder. You should only
do this when the turbulence box size changes (e.g., if the rotor size changes)
or if you have a model that's never been simulated on mimer.
2. Determine your set ID code. There are two scenarios:
    * No one else in your project has run simulations on mimer. In this case,
    create a new set ID alpha code (e.g., "AA", "AB", etc.).
    * Simulations for this project/turbine configuration already exist. In this
    case, use a pre-existing set ID alpha code and add one to the most recent
    Set ID (e.g., if "AB0008" exists, your new folder should be "AB0009").

##### 3. Add proper log files for your Set ID folder.
See the [house rules](docs/houserules-mimerhawc2sim.md) regarding log files.

##### 4. Add your model files.
Within your new Set ID folder, add your HAWC2 model files. Keep a folder
structure similar to this:

```
|-- control/
|   |-- ...
|-- data/
|   |-- ...
|-- htc/
|   |-- _master/
|   |   |-- TURB_master_AA0001.htc
|   |-- DLCs.xlsx
```

Your master htc file, stored in ```htc/_master/```, can take any desired naming
convention, but it must have ```_master_``` in the name or future scripts will
abort. ```htc/DLCs.xlsx``` is your master Excel file that will create the
subordinate Excel files in the coming steps.

##### 5. Create your subordinate Excel files.
From a terminal, change to your htc directory. Then run the following code:

```
$ export PATH=/home/python/miniconda3/bin:$PATH
$ source activate py36-wetb
$ python /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/GenerateDLCs.py --folder=DLCs
$ source deactivate
```

This will create a subfolders DLCs and fill that new subfolder with the created
subordinate Excel files.

##### 6. Create your htc files and PBS job scripts .
These files and scripts are generated from the subordinate Excel files from
Step 5. To do this, in the terminal, change up a level to your Set ID folder
(e.g., to folder "AB0001"). Then run this code

```
$ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --prep
```

Your htc files should now be placed in subfolders in the htc folder, and PBS
job files should be in folder ```pbs_in```.

##### 7. Launch the htc files to the cluster.
Use the ```launch.py``` function to launch the jobs on the cluster.
For example, the following code will launch the jobs in folder ```pbs_in``` on
100 nodes. You must be in the top-level Set ID folder for this to work (e.g.,
in folder "AB0001").

```
$ launch.py -n 100 -p pbs_in/
```

There are many launch options available. You can read more about the options
and querying the cluster configurations/status/etc. on
[this page](docs/howto-make-dlcs.md), or you can use the ```launchy.py```
help function to print available launch options:

```
$ launch.py --help
```

##### 8. Post-process results.

The wetb function ```qsub-wrap.py``` can not only generate htc files but also
post-process results. For example, here is code to check the log files
and calculate the statistics, the AEP and the lifetime equivalent loads
(must be executed from the top-level Set ID folder):

```
$ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --years=25 --neq=1e7 --stats --check_logs --fatigue
```

More details regarding loading the post-processed with statistics dataframes
can be found here: [using-statistics-df](docs/using-statistics-df.md).

