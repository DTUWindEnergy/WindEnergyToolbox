
[![build status](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/badges/master/build.svg)](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/commits/master)
[![coverage report](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/badges/master/coverage.svg)](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/commits/master)
[![pypi status](https://img.shields.io/pypi/v/wetb.png)](https://pypi.python.org/pypi/wetb)

# Introduction

The Wind Energy Toolbox (or ```wetb```, pronounce as wee-tee-bee) is a collection
of Python scripts that facilitate working with (potentially a lot) of HAWC2,
HAWCStab2, FAST or other text input based simulation tools.

Note that this toolbox is very much a WIP (work in progress). For example,
some of the functions in the [prepost](#prepost) module have a similar functions
in [Hawc2io](wetb/hawc2/Hawc2io.py). These different implementations will be
merged in due time.

Both Python2 and Python3 are supported.

# Installation

See documentation.

# Documentation

[https://toolbox.pages.windenergy.dtu.dk/WindEnergyToolbox/](https://toolbox.pages.windenergy.dtu.dk/WindEnergyToolbox/)


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
- [ae_file](wetb/hawc2/ae_file.py): Read/write/manipulate AE (aerodynamic blade layout) files
- [pc_file](wetb/hawc2/pc_file.py): Read/write/manipulate PC (profile coefficient) files
- [st_file](wetb/hawc2/st_file.py): Read/write/manipulate ST (structural properties) files
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

Additional documentation can be found here:

- [Getting started with DLBs](docs/getting-started-with-dlbs.md)
    - [Generate DLB spreadsheets](docs/generate-spreadsheet.md)
    - [Auto-generation of Design Load Cases](docs/howto-make-dlcs.md)
    - [House rules for storing results on ```mimer/hawc2sim```](docs/houserules-mimerhawc2sim.md)
    - [How to use the Statistics DataFrame](docs/using-statistics-df.md)

### [fast](wetb/fast)
Tools for working with NREL's FAST code (An aeroelastic computer-aided engineering (CAE) tool for horizontal axis wind turbines)
- [fast_io](wetb/fast/fast_io.py): Read binary and ascii result files

### [utils](wetb/utils)
Other functions
- [geometry](wetb/utils/geometry.py): Different kind of geometry conversion functions
- [process_exec](wetb/utils/process_exec.py): Run system command in subprocess
- [timing](wetb/utils/timing.py): Decorators for evaluating execution time of functions
- [caching](wetb/utils/caching.py): Decorators to create cached (calculate once) functions and properties

