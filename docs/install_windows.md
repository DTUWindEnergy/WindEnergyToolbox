# Install wetb on Windows

For updating the toolbox jump to [Update wetb on Windows](#Update wetb on Windows)

This guide describes a simple way to install the toolbox on Windows (1/9-2016)

### Installing Anaconda

* Anaconda is a professional grade, full blown scientific Python distribution.

* Download and install Anaconda (Python 3.5 version, 64 bit installer) from <https://www.continuum.io/downloads>


* Update the root Anaconda environment (type in a terminal):

```
conda update --all
```

### Create Anaconda environment

* Creating an environment :

```
conda create -n py35 python=3.5
```

* Activate the envirronment:

```
activate py35
```

The python distribution in use will now be located in \<path_to_anaconda\>/env/py35/


### Install dependencies with conda and pip

* Install the necessary Python dependencies using the conda package manager:

```
conda install setuptools_scm future h5py pytables pytest nose sphinx
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl
```

* Not all packages are available in the conda repositories, but they can be
easily installed with pip:

```
pip install pyscaffold pytest-cov --no-deps
```


## And Finally: install wetb

```
pip install wetb
```

# Update wetb on Windows

* Activate the envirronment (type in a terminal):

```
activate py35
´´´

* Update the toolbox

```
pip install wetb --upgrade --no-deps
´´´
