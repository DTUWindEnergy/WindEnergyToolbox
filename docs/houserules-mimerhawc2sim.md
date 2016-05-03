
House Rules for ```mimer/hawc2sim``` and HAWC2 model folder structure
=====================================================================


Objectives
----------

* Re-use turbulence boxes (save disk space)
* Find each others simulations, review, re-run
* Find working examples of simulations, DLB's
* Avoid running the same DLB, simulations more than once

* Disk usage quota review: administrators will create an overview of disk usage
as used per turbine and user.


Basic structure
---------------

The HAWC2 simulations are located on the data capacitor [mimer]
(http://mimer.risoe.dk/mimerwiki), on the following address:

```
# on Windows, use the following address when mapping a new network drive
\\mimer\hawc2sim

# on Linux you can use sshfs or mount -t cifs
//mimer.risoe.dk/hawc2sim
```

The following structure is currently used for this ```hawc2sim``` directory:
* turbine model (e.g. DTU10MW, NREL5MW, etc)
    * set ID: 2 alphabetic characters followed by 4 numbers (e.g. AA0001)
* letters are task/project oriented, numbers are case oriented

For example:
* DTU10MW
    * AA0001
    * AA0002
    * AB0001
    * log_AA.xlsx
    * log_BB.xlsx
    * log_overview.xlsx
* NREL5MW
    * AA0001
    * AA0002
    * BA0001
    * log_AA.xlsx
    * log_BB.xlsx
    * log_overview.xlsx


House rules
-----------

* New Turbine model folder when a new size of the turbulence box is required
(i.e. when the rotor size is different)
* One "set ID" refers to one analysis, and it might contain more than one DLB
	* If you realize more cases have to be included, add them in the same
	"set ID". Don't start new "set ID" numbers.
	* Each "set ID" number consists of 2 alphabetic followed by 4
	numerical characters.
* Log file
	* Indicate which DLB used for the given "set ID" in the log file
	* Indicate the changes wrt to a previous "set ID"
	* Write clear and concise log messages so others can understand what
	analysis or which DLB is considered
	* Indicate in the log if something works or not.
	* Indicate if a certain "set ID" is used for a certain publication or report
* Keep a log file of the different letters. For instance AA might refer to load
simulations carried out within a certain project
* When results are outdated or wrong, delete the log and result files, but keep
the htc, data and pbs input files so the "set ID" could be re-run again in the
future. This is especially important if the given "set ID" has been used in a
publication, report or Master/PhD thesis.


File permissions
----------------

* By default only the person who generated the simulations within a given
"set ID" can delete or modify the input files, other users have only read access.
If you want to give everyone read and write access, you do:

```
# replace demo/AA0001 with the relevant turbine/set id
g-000 $ cd /mnt/mimer/hawc2sim/demo
g-000 $ chmod 777 -R AA0001
```


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
master files and DLC definition spreadsheet files on ```mimer/hawc2sim```.

