
# Introduction

The Wind Energy Toolbox (or ```wetb```, pronounce as wee-tee-bee) is a collection
of Python scripts that facilitate working with (potentially a lot) of HAWC2,
HAWCStab2, FAST or other text input based simulation tools.

Note that this toolbox is very much a WIP (work in progress). For example,
some of the functions in the [prepost](#prepost) module have a similar functions
in [Hawc2io](wetb/hawc2/Hawc2io.py). These different implementations will be
merged in due time.


# Python 3

This module currently only works under Python 3. If you are working in Python 2,
this could be a good moment to consider switching. If you are bound to Python 2
due to critical 3th party dependencies you are encouraged to cast your vote for
Python 2 compatibility in
[issue 1](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues/1).

Switching to Python 3 is in general a very good idea especially since Python 3.5
was released. Some even dare to say it
[is like eating your vegetables](http://nothingbutsnark.svbtle.com/porting-to-python-3-is-like-eating-your-vegetables).

You can automatically convert your code from Python 2 to 3 using the
[2to3](https://docs.python.org/2/library/2to3.html) utility which is included
in Python 2.7 by default. You can also write code that is compatible with both
2 and 3 at the same time (you can find additional resources in
[issue 1](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues/1)).


# Dependencies

* [numpy](http://www.numpy.org/)

* [scipy](http://scipy.org/scipylib/)

* [pandas](http://pandas.pydata.org/)

* xlrd

* h5py

* [matplotlib](http://matplotlib.org/)

* [pytables](http://www.pytables.org/)

* [pyscaffold](http://pyscaffold.readthedocs.org/en/)

* pytest

* six


# Installation

Detailed installation instructions, including how to install Python from scratch,
are described in the [detailed installation manual](docs/install-manual-detailed.md).


If you now what you are doing, you can install as follows:

```
python setup.py
```

Or create a binary wheel distribution package with:

```
python setup.py bdist_wheel -d dist
```


# Tests

Only a small part of the code is covered by unittests currently. More tests are
forthcoming.


# Contents of WindEnergyToolbox, [wetb](wetb)

### Overview

- [hawc2](#hawc2)
- [gtsdf](#gtsdf)
- [fatigue_tools](#fatigue_tools)
- [wind](#wind)
- [dlc](#dlc)
- [prepost](#prepost)
- [fast](#fast)
- [utils](#utils)

### [hawc2](wetb/hawc2)
- [Hawc2io](wetb/hawc2/Hawc2io.py): Read binary, ascii and flex result files
- [sel_file](wetb/hawc2/sel_file.py): Read/write *.sel (sensor list) files
- [htc_file](wetb/hawc2/htc_file.py): Read/write/manipulate htc files
- [ae_file](wetb/hawc2/ae_file.py): Read AE (aerodynamic blade layout) files
- [pc_file](wetb/hawc2/pc_file.py): Read PC (profile coefficient) files
- [shear_file](wetb/hawc2/shear_file.py): Create user defined shear file
- [at_time_file](wetb/hawc2/at_time_file.py): Read at output_at_time files
- [log_file](wetb/hawc2/log_file.py): Read and interpret log files
- [ascii2bin](wetb/hawc2/ascii2bin): Compress HAWC2 ascii result files to binary

### [gtsdf](wetb/gtsdf)
General Time Series Data Format, a binary hdf5 data format for storing time series data.
- [gtsdf](wetb/gtsdf/gtsdf.py): read/write/append gtsdf files
- [unix_time](wetb/gtsdf/unix_time.py): convert between datetime and unix time (seconds since 1/1/1970)

### [fatigue_tools](wetb/fatigue_tools)
- [fatigue](wetb/fatigue_tools/fatigue.py): Rainflow counting, cycle matrix and equivalent loads
- [bearing_damage](wetb/fatigue_tools/bearing_damage.py): Calculate a comparable measure of bearing damage

### [wind](wetb/wind)
- [shear](wetb/wind/shear.py): Calculate and fit wind shear 

### [dlc](wetb/dlc)
Module for working with "Design load cases" (Code independent)
- [high_level](wetb/dlc/high_level.py) Class for working with the highlevel dlc excell sheet

### [prepost](wetb/prepost)
Module for creating an arbitrary number of HAWC2 simulations, and optionally
corresponding execution scripts for a PBS Torque cluster (Linux), simple bash
(Linux), or Windows batch scripts. A post-processing module is also included
that calculates statistical parameters, performs rainflow counting for fatigue
load calculations, and create load envelopes.

### [fast](wetb/fast)
Tools for working with NREL's FAST code (An aeroelastic computer-aided engineering (CAE) tool for horizontal axis wind turbines)
- [fast_io](wetb/fast/fast_io.py): Read binary and ascii result files

### [utils](wetb/utils)
Other functions
- [geometry](wetb/utils/geometry.py): Different kind of geometry conversion functions
- [process_exec](wetb/utils/process_exec.py): Run system command in subprocess
- [timing](wetb/utils/timing.py): Decorators for evaluating execution time of functions
- [caching](wetb/utils/caching.py): Decorators to create cached (calculate once) functions and properties


# Note

This project has been set up using PyScaffold 2.5. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.

