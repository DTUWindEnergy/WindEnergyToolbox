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
[docs/using-statistics-df.md](using-statistics-df) has some information
on loading the post-processing statistics using Python.


## Steps

##### 1. Make sure that you can access the cluster/mimer according to the instructions in [wetb](README.md)

##### 2. Create a Set ID folder for this project/simulation (detailed house rules [houserules-mimerhawc2sim](docs/houserules-mimerhawc2sim.md))
a. Navigate to or create the folder corresponding to your model on
```\\mimer.risoe.dk\hawc2sim```. You should create a new turbine model folder
only when the turbulence box size changes (e.g., if the rotor size changes) or
if you have a completely new model. You now have two options:

    i. Create a new Set ID alpha code (e.g., "AA", "AB", etc.) You should
    only do this if no one else in your project has run simulations on mimer.

    ii. Use a pre-existing Set ID alpha code. This is preferable, but be
    sure to contact whoever created/used the alpha code most recently to
    verify it is okay to use.

b. Create a new Set ID folder. If you have a new Set ID alpha code (e.g.,
"AB"), then it should be that code followed by "0000". If you are using a
pre-existing Set ID alpha code, then just add 1 to the most recent number
(e.g., if "AB0008" exists, your new folder is "AB0009").

##### 3. Add proper log files for your Set ID folder. (See [using-statistics-df](docs/using-statistics-df.md))

##### 4. Add your model files.
Within your new Set ID folder, add your HAWC2 model files. Keep a folder
structure similar to this:

```
|-- animation
|-- control
|-- data
|-- externalforce
|-- htc
|-- iter
|-- logfiles
|-- pbs_in
|-- pbs_out
|-- res
|-- res_eigen
|-- turb
AA0001_ErrorLog.csv
DTU10MW_AA0001.zip
pbs_in_file_cache.txt
```

Your ```htc```, ```pbs_in```, and ```pbs_out``` folders will be empty at this time.

##### 5. Structure your Set ID/htc folder like this:
```
|-- _master
DLCs.xlsx
```

Place your master HTC file in the _master folder (NOTE that it must have
"_master_" in its name or later scripts won’t find it). ```DLCs.xlsx``` is
your master Excel file that will create the subordinate Excel files in the coming
steps.

##### 6. Create your subordinate Excel files.
From a terminal, change to your htc directory. Then run the following code:

```
$ export PATH=/home/python/miniconda3/bin:$PATH
$ source activate wetb_py3
$ python /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/GenerateDLCs.py --folder=DLCs
$ source deactivate
```

This will create a subfolders DLCs and fill that new subfolder with the created
subordinate Excel files.

##### 7. Move your DLCs.xlsx file from the htc folder to the _master folder.
It will cause errors in later scripts if left in the htc folder.

##### 8. Create your htc files and corresponding PBS job scripts from the subordinate Excel files.
In the terminal, change up a level to your Set ID folder (e.g., to folder
"AB0001"). Then run this code

```
$ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --prep
```

Your htc files should now be placed in subfolders in the htc folder, and PBS
job files should be in folder ```pbs_in```.

##### 9. Launch the htc files to the cluster.
For example, the following code will launch the jobs in folder pbs_in on 100
nodes. You must be in the top-level Set ID folder for this to work (e.g., in
folder "AB0001").

```
$ launch.py -n 100 -p pbs_in/
```

use the ```launchy.py``` help function to get more options:

```
$ launch.py --help
```

More information on launching options and cluster configurations/status/etc.
can be found on the [wetb README](README.md).

##### 10. Post-process results.
More detail in [docs/using-statistics-df.md](using-statistics-df). Here is
example code to check the log files, calculate the statistics, the AEP and the
lifetime equivalent loads: (Execute from top-level Set ID folder)

```
$ qsub-wrap.py -f /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/dlctemplate.py --years=25 --neq=1e7 --stats --check_logs --fatigue
```

