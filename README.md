# Contents of WindEnergyToolbox, [wetb](wetb)

- [hawc2](#hawc2)
- [gtsdf](#gtsdf)
- [fatigue_tools](#fatigue_tools)
- [wind](#wind)
- [dlc](#dlc)
- [fast](#fast)
- [utils](#utils)

------------------------------------------------------------------------------------
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
- [fatigue](wetb/fatigue_tools/fatigue.py): Rainflow counting, cycle matrix and equvivalent loads
- [bearing_damage](wetb/fatigue_tools/bearing_damage.py): Calculate a comparable measure of bearing damage

### [wind](wetb/wind)
- [shear](wetb/wind/shear.py): Calculate and fit wind shear 

### [dlc](wetb/dlc)
Module for working with "Design load cases" (Code independent)
- [high_level](wetb/dlc/high_level.py) Class for working with the highlevel dlc excell sheet


### [fast](wetb/fast)
Tools for working with NREL's FAST code (An aeroelastic computer-aided engineering (CAE) tool for horizontal axis wind turbines)
- [fast_io](wetb/fast/fast_io.py): Read binary and ascii result files

### [utils](wetb/utils)
Other functions
- [geometry](wetb/utils/geometry.py): Different kind of geometry conversion functions
- [process_exec](wetb/utils/process_exec.py): Run system command in subprocess
- [timing](wetb/utils/timing.py): Decorators for evaluating execution time of functions
- [caching](wetb/utils/caching.py): Decorators to create cached (calculate once) functions and properties