Auto-generation of DLB Spreadsheets
===================================

Introduction
------------

This manual explains how to automatically generate the set of spreadsheets that
defines a DLB and is required as input to the pre-processor.

This tool comes handy in the following scenarios:
* a DLB for a new turbine needs to be generated;
* a different wind turbine class needs to be evaluated;
* a new parameter needs to be included in the htc file;
* different parameters variations are required, e.g. different wind speed range or different number of turbulent seed.

The generator of the cases uses an input spreadsheet where the cases are defined
in a more compact way.
The tool is based on the "tags" concept that is used for the generation of the htc files.

Main spreadsheet
----------------

A main spreadsheet is used to defines all the DLC of the DLB. The file specifies the tags that are then required in the htc files.

The file has:
* a Main sheet where some wind turbines parameters are defined, the tags are initialized, and the definitions of turbulence and gusts are given.
* a series of other sheets, each defining a DLC. In these sheets the tags that changes in that DLC are defined.

The tags are divided into three possible different categories:
* Constants (C). Constants are tags that do not change in a DLC, e.g. simulation time, output format, ...;
* Variables (V). Variables are tags that define the number of cases in a DLC through their combinations, e.g. wind speed, number of turbulence seeds, wind direction, ..;
* Functions (F). Functions are tags that depend on other tags through an expression, e.g. turbulence intensity, case name, ....

In each sheet the type of tag is defined in the line above the tag by typing one of the letters C, V, or F.

Functions (F) tags
------------------

* Numbers can be converted to strings (for example when a tag refers to a file name)
by using double quotes ```"``` for Functions (F):
    * ```"wdir_[wdir]deg_wsp_[wsp]ms"``` will result in the tags ``` [wdir]```
    and ```[wsp]```  being replaced with formatted text.
    * following formatting rules are used:
        * ```[wsp]```, ```[gridgustdelay]``` : ```02i```
        * ```[wdir]```, ```[G_phi0]``` : ```03i```
        * ```[Hs]```, ```[Tp]``` : ```05.02f```
        * all other tags: ```04i```
    * Only numbers in tags with double quotes are formatted. In all other cases
    there is no formatting taking place and hence no loss of precision occurs.
    * In this context, when using quotes, always use double quotes like ```"```.
    Do not use single quotes ```'``` or any other quote character.

Variable (V) tags
-----------------

* ```[seed]``` and ```[wave_seed]``` are special variable tags. Instead of defining
a range of seeds, the user indicates the number of seeds to be used.
* ```[wsp]``` is a required variable tag
* ```[seed]``` should be placed in a column BEFORE ```[wsp]```

Generate the files
------------------

To generate the files defining the different DLC the following lines need to be executed:

    export PATH=/home/python/miniconda3/bin:$PATH
    source activate py36-wetb
    python /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/GenerateDLCs.py --folder=DLCs

the first two lines activate the virtual environment. The third calls the routine *GenerateDLCs.py * that generates the files.
The routine should be called from the folder *htc* where also the master spreadsheet *DLCs.xlsx* need to be located.
The generated files are placed in the folder *DLCs*.
