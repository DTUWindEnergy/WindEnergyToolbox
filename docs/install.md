# Anaconda (Windows/Mac/Linux)

## Installation

Install the necessary Python dependencies using the ```conda``` package manager:

```
>> conda install setuptools_scm mock h5py pytables pytest pytest-cov nose sphinx blosc pbr paramiko
>> conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl psutil
>> conda install -c conda-forge sshtunnel --no-deps
```

Now you can install ```wetb``` with ```pip``` (there is no ```conda``` package
available yet, see [issue 21](toolbox/WindEnergyToolbox#21)).
Since we prefer that ```conda``` manages and installs all dependencies we
expclicitally tell ```pip``` to only install ```wetb``` and nothing more:

```
>> pip install wetb --upgrade --no-deps
```

## Update conda and ```wetb```

```
>> conda update --all
>> pip install wetb --upgrade --no-deps

```


# Pip (Windows/Mac/Linux)

Do not use this procedure in conda environments. See above.


## Installation and update

```
>> pip install --upgrade wetb
```

