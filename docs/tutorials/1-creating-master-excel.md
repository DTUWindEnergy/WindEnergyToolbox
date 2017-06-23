# Tutorial 1: Creating master Excel file

The Wind Energy Toolbox has a workflow for automatically running design load 
bases (DLBs) on Gorm.
This workflow has the following steps:
1. Create a master Excel sheet defining each case in the DLB
2. [Create subordinate Excel sheets from each tab in the master Excel sheet](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/tutorials/2-creating-subordinate-excels.md)
3. [Create htc files and PBS job scripts for each requisite simulation using 
the subordinate Excel files and a master htc file.](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/tutorials/3-creating-htc-pbs-files.md)
4. Submit all PBS job scripts to the cluster
5. Post-process results
6. Visualize results

This tutorial presents how to accomplish Step 1.

Note that it is possible to customize your simulations by skipping/modifying 
steps.
Such a procedure will be discussed in a later tutorial.

If there are any problems with this tutorial, please [submit an issue](
https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues).

## 1. Background: Master Excel File

The master Excel file is an Excel file that is used to create subordinate 
Excel files for generation of htc files and PBS job scripts.

### Master file structure

The master Excel file has a main tab, called "Main", that defines default 
values  and necessary functions that are called in the other tabs.
Each other tab defines a new case, and one subordinate Excel file will be 
generated for each case.
There are three variable types in the master Excel file:
- Constants: values that do not change within a case
- Variables: values that do change within a case, but are numbers that do not 
depend on any other values (e.g., wind speed in DLC 1.2)
- Functions: values that depend on other values

### Tag names

The values that are defined in the master Excel file (and eventually the 
subordinate Excel files) are used to replace "tags" in the master htc file.
These tags are of the form ```[$TAG_NAME]```.
Theoretically, a user can define any new tags they desire, there are no 
require naming conventions.
However, there are some tags that are currently hard-coded into the Toolbox 
that can result in errors if the tag names are changed.
Thus, **we do not recommend you change the tag names from those in the 
tutorial**.
If you need new values that do not exist in the tutorial's master htc file 
and produced master file, then it should be fine to add them.

There are a few tags that deserve special mention:
- ```[Case folder]```: the htc files for each case will be saved in this case 
folder. We do not recommend changing the tag name or naming convention here 
if you are doing a standard DLB.
- ```[Case id.]```: this defines the naming convention for each htc file. We 
do not recommend changing the tag name or naming convention here if you are 
doing a standard DLB.
- ```[seed]```: this variable indicates the desired number of seeds for each 
set of variables. Thus, for example, in DLC 1.2, 1.3, the ```[seed]``` value 
should be set to at least 6.

Lastly, it is extremely important that your tag names in your master Excel 
file match the tag names in your master htc file.
Thus, **be sure to verify that your tag names in your master Excel and master 
htc files are consistent**.

## 2. Tutorial

The procedure for creating the master Excel sheet is simple: each desired DLB 
is defined in a tab-delimited text file, and these are loaded into a single 
Excel file.
It is assumed that the user has a collection of text files in a folder for 
all of the DLBs to be simulated.
This tutorial uses the text files located in 
```wetb/docs/tutorials/data/DLCs_onshore```, which contain a series of text 
files for a full DLB of an onshore turbine.
These text files correspond to the onshore DTU 10 MW master htc file that is 
located in the same directoy.

Generate the master Excel file in a few easy steps:
1. Open a command window.
2. If you are running the tutorial locally (i.e., not on Gorm), navigate to 
the Wind Energy Toolbox tutorials directory.
3. From a terminal/command window, run the code to generate the Excel file 
from a folder of text files:
    * Windows (from the wetb tutorials folder):  
    ```python ..\..\wetb\prepost\write_master.py --folder data\DLCs_onshore --filename DLCs_onshore.xlsx```
    * Mac/Linux (from the wetb tutorials folder):  
    ```python ../../wetb/prepost/write_master.py --folder data/DLCs_onshore  --filename DLCs_onshore.xlsx```
    * Gorm (from any folder that contains a subfolder with your text files. Note
you must activate the wetb environment (see Step 5 [here](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/blob/master/docs/getting-started-with-dlbs.md)
) before this command will work. This command also assumes the folder with your
text files is called "DLCs_onshore" and is located in the working directory.):  
    ```python  /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/write_master.py --folder ./DLCs_onshore --filename ./DLCs_onshore.xlsx```
 
The master Excel file "DLCs_onshore.xlsx" should now be in the your current 
directory.

Note that we have used the parser options ```--folder``` and ```--filename``` 
to specify the folder with the text files and the name of the resulting Excel 
file.
Other parser options are also available.
(See doc string in ```write_master.py``` function.)

## 3. Generation options

See doc string in ```write_master.py``` function.

## 4. Issues

If there are any problems with this tutorial, please [submit an issue](
https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues).
We will try to fix it as soon as possible.

