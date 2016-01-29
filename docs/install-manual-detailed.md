
!! This guide is not finished yet, it is a WIP (Work In Progress) !!

# Detailed Installation Manual

Installing Python packages with compiled extensions can be a challenge especially
on Windows systems. However, when using Miniconda things can be simplified to a
great extent as this manual hopefully will show you.

The this approach will require you to use the command line, but it is as easy
as copy-pasting them from this page straight into your command prompt.


## Using Miniconda

* Download the latest Python 3 (!!) Miniconda installer for your platform
[here](http://conda.pydata.org/miniconda.html)

* No need to worry about Python 2 or 3 at this stage. You can still use the
Python 3 installer for creating Python 2 conda environments

* Install the necessary Python dependencies using the conda package manager:

```
conda install scipy pandas matplotlib cython xlrd pytables sphinx mingw
```

* Not all packages are available in the conda repositories, but they can be
easily installed with pip:

```
pip install pyscaffold pytest pytest-cov
```

