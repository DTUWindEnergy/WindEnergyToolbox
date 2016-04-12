
House Rules for ```mimer/hawc2sim```
====================================


Objectives
----------

* Re-use turbulence boxes (save disk space)
* Find each others simulations, review, re-run
* Find working examples of simulations, DLB's
* Avoid running


House rules
-----------

* New Turbine model folder when a new size of the turbulence box is required
(so basically when the rotor size is different)
* One set ID refers to one analysis, and it might contain more than one DLB
	* If you realize more cases have to be included, just add them in the same
	set ID. Don't start new set ID numbers.
* Log file
	* Indicate which DLB used for the given "set ID" in the log file
	* Indicate the changes wrt to a pervious set ID
	* Write clear log messages so others can understand what is going on
	* Indicate in the log if something works or not.
	* Indicate if a certain set ID is used for a certain publication or report
* Keep a log file of the different letters
* Disk usage quota review
* When results are outdated or wrong, delete the log and result files, but keep
the htc, data and pbs input files so the set ID could be run again in the
future. This is especially important if the given set ID has been used in a
publication, report or Master/PhD thesis.


HAWC2 folder structure
----------------------

The current DLB setup assumes the following HAWC2 model folder structure:

```
|-- control
|   |-- ...
|-- data
|   |-- ...
|-- htc
|   |-- DLCs
|   |   |-- dlc12_iec61400-1ed3.xlsx
|   |   |-- dlc13_iec61400-1ed3.xlsx
|   |   |-- ...
|   |-- _master
|   |   `-- dtu10mw_master_C0013.htc
```

The load case definitions should be placed in Excel spreadsheets with a
```*.xlsx``` extension. The above example shows one possible scenario whereby
all the load case definitions are placed in ```htc/DLCs``` (all folder names
are case sensitive). Alternatively, one can also place the spreadsheets in
separate sub folders, for example:

```
|-- control
|   |-- ...
|-- data
|   |-- ...
|-- htc
|   |-- dlc12_iec61400-1ed3
|   |   |-- dlc12_iec61400-1ed3.xlsx
|   |-- dlc13_iec61400-1ed3
|   |   |-- dlc13_iec61400-1ed3.xlsx
```

In order to use this auto-configuration mode, there can only be one master file
in ```_master``` that contains ```_master_``` in its file name.

For the NREL5MW and the DTU10MW HAWC2 models, you can find their respective
master files and DLC definition spreadsheet files on Mimer. When connected
to Gorm over SSH/PuTTY, you will find these files at:
```
/mnt/mimer/hawc2sim # (when on Gorm)
```

