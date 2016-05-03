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
The tool is based on the "tags" concept that is used for the genetaion of the htc files.

Main spreatsheet
----------------

A main spreadsheet is used to defines all the DLC of the DLB. The file specifies the tags that are then required in the htc files.

The file has:
* a Main sheet where some wind turbines parameters are defined, the tags are initialized, and the definitions of turbulence and gusts are given.
* a series of other sheets, each defining a DLC. In these sheets the tags that changes in that DLC are defined.

The tags are devided into three possible different categories:
* Constants (C). Constants are tags that do not change in a DLC, e.g. simulation time, output format, ...;
* Variables (V). Variables are tags that define the number of cases in a DLC through their combinations, e.g. wind speed, number of turbilence seeds, wind direction, ..;
* Functions (F). Functions are tags that depend on other tags through an expression, e.g. turbulence intensity, case name, ....

In each sheet the type of tag is defined in the line above the tag by typing one of the letters C, V, or F.

Generate the files
------------------

To generate the files defining the different DLC the following lines need to be executed:
    
    export PATH=/home/python/miniconda3/bin:$PATH
    source activate wetb_py3
    python /home/MET/repositories/toolbox/WindEnergyToolbox/wetb/prepost/GenerateDLCs.py --folder=DLCs 
    
the first two lines activate the virtual environment. The third calls the routine *GenerateDLCs.py * that generates the files.
The routine should be called from the folder *htc* where also the master spreadsheet *DLCs.xlsx* need to be located.
The generated files are placed in the folder *DLCs*.
