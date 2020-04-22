# Tutorial 2: Creating subordinate Excel files

The Wind Energy Toolbox has a workflow for automatically running design load
bases (DLBs) on Gorm.
This workflow has the following steps:
1. [Create a master Excel sheet defining each case in the DLB](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/tutorials/1-creating-master-excel.md)
2. Create subordinate Excel sheets from each tab in the master Excel sheet
3. [Create htc files and PBS job scripts for each requisite simulation using
the subordinate Excel files and a master htc file.](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/tutorials/3-creating-htc-pbs-files.md)
4. Submit all PBS job scripts to the cluster
5. Post-process results
6. Visualize results

This tutorial presents how to accomplish Step 2.

Note that it is possible to customize your simulations by skipping/modifying
steps.
Such a procedure will be discussed in a later tutorial.

If there are any problems with this tutorial, please [submit an issue](
https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues).

## 1. Background: Subordinate Excel Files

The subordinate Excel files are a series of basic Excel files that are
generated from the master Excel file. (See our tutorial on generating the
master Excel file [here]( https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/tut orials/1-creating-master-excel.md).)
There is a different subordinate Excel file for every tab in the master Excel
file, except for the "Main" tab, one for each case to simulate (e.g., design
load case 1.2 from IEC-61400-1).
Each subordinate Excel file has a single tab that lists the different tag
values for the htc master file in the column, and each row corresponds to a
different htc file to be generated.
The generation of the htc files from the subordinate Excel files is discused
in the next tutorial.


## 2. Tutorial

The generation of the subordinate Excel files is done using the
[GenerateDLS.py](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/wetb/prepost/GenerateDLCs.py)
function in the Wind Energy Toolbox.
On Gorm, the command can be executed from the htc directory with the master
Excel file as follows:
```
export PATH=/home/python/miniconda3/bin:$PATH
source activate py36-wetb
python /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/GenerateDLCs.py [--folder=$FOLDER_NAME] [--master=$MASTER_NAME]
source deactivate
```

The ```export PATH``` command adds the miniconda bin directory to the path,
which is necessary for the toolbox.
The ```source activate py36-wetb``` and ```source deactivate``` are
Gorm-specific commands to activate the Wind Energy Toolbox Python environment.
The ```--folder``` and ```--master``` flags are optional flags to specify,
respectively, the name of the folder to which the subordinate Excel files
should be written to and the name of the master Excel file.
The default values for these two options are './' (i.e., the current
directory) and 'DLCs.xlsx', respectively.

After running the commands in the above box on Gorm, you should see a folder in
your htc directory with all of your subordinate Excel files.


## 3. Issues

If there are any problems with this tutorial, please [submit an issue](
https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues).
We will try to fix it as soon as possible.

