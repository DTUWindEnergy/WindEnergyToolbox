
!! This guide is not finished, and might contain innacuracies. Please report
any mistakes/bugs by creating an
[issue](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues).
This is a WIP (Work In Progress) !!

# Detailed Installation Manual

Installing Python packages with compiled extensions can be a challenge especially
on Windows systems. However, when using Miniconda things can be simplified to a
great extent as this manual hopefully will show you.

The this approach will require you to use the command line, but it is as easy
as copy-pasting them from this page straight into your command prompt.

Installation instructions follow in alphabetical orderby platorm.


## Linux

* Use either your system package manager, pip + virtualenv, or Anaconda to
install the following python dependencies:

> numpy, cython, scipy, pandas, matplotlib, pyscaffold, future, nose, sphinx,
> xlrd, (py)tables, h5py, pytest, pytest-cov, setuptools_scm, setuptools

Note that often the pytables packages is called python-tables instead of
python-pytables.

* Other tools you will need:

> git gcc


## Mac

People who now how to handle Python on the Mac side are kindly requested to
complete this guide :-)


## Windows

A Python installation with compilers is required. If you already have this,
or know how set up such an environment, you skip to
[here](install-manual-detailed.md#and-finally-install-wetb).


### Microsft Visual Studio 2010


### Command line

This guide will use the command line (aka command prompt) frequently.
You can launch a Windows terminal as follows: press Start> and type
"cmd" + <Enter>. A link to the command prompt should be visible now.

In case you want an alternative, more capable windows terminal, you could consider
using [ConEmu](https://conemu.github.io/) (this is optional).

> ConEmu-Maximus5 is a Windows console emulator with tabs, which presents
> multiple consoles and simple GUI applications as one customizable GUI window
> with various features.


### Git

* Download and install Git version control system for Windows 64-bit
[here](https://git-scm.com/download/win). Only select the Windows Portable
option if you know what you are doing or if you do not have administrative
rights on your computer.

* Git comes with a simple GUI, but there are more and different options available
if you are not happy with it, see [here](https://git-scm.com/downloads/guis).

* If you would like to use a GUI for git, we recommend you to use
[tortoisegit](https://tortoisegit.org/)


### Option 1: Anaconda (large download)

* Anaconda is a professional grade, full blown scientific Python distribution.

* Download Anaconda Python 2.7 for Windows [here](https://www.continuum.io/downloads).
Depening on whether you already have a Python system installation, or another
Anaconda version, you can should chose wisely between the two offered options:

    * Add Anaconda to the system PATH: you can only have one Anaconda installation
added to your path, but you can have multiple Anaconda or Miniconda installations
in parallelel.

    * Set as your Python default system installation (you can only have one Python
installation set as your systems default).


* Although the ```wetb``` module is Python 3.5 compatible, compiling it under
Windows requires MS Visual Studio 2015. When using Visual Studio 2010, you can
compile extensions for both Python 2.7 and Python 3.4. If you want to use
Python 3+, we recommend to use 3.4 for the time being, unless if you know what
you are doing (and wouldn't care to share installation instructions):

* Update the root Anaconda environment:

```
conda update --all
```

* You can now create an independent environment with a specific python version:

```
conda create -n py27 python=2.7
conda create -n py34 python=3.4
```

* These environments can be activated as follows:

```
activate py27
activate py34
```

use ```deactivate``` to deactivate the environment.


* You can only have the full Anaconda distribution in the root environment.
When creating new environments you will have to manually install all packages
you require for that specific environment.


### Option 2: Miniconda (smaller download)

* MS Visual Studio 2010

* For building Python packages with Python 2.7,
[http://aka.ms/vcpython27](http://aka.ms/vcpython27), here's the
[direct link](https://www.microsoft.com/en-gb/download/details.aspx?id=44266).

* Download the latest Python 3 (!!) Miniconda installer for your platform
[here](http://conda.pydata.org/miniconda.html)

* No need to worry about Python 2 or 3 at this stage. You can still use the
Python 3 installer for creating Python 2 conda environments

* Before creating/activation/updating any specific miniconda environemnt,
update the global miniconda environment (conda, pip, etc)

```
conda update --all
```

* Create a a new environment
```
conda create -n py27 python=2.7
conda create -n py34 python=3.4
```

* Activate the envirnment

```
activate py27
```


### Install dependencies with conda and pip

* Install the necessary Python dependencies using the conda package manager:

```
conda install scipy pandas matplotlib cython xlrd pytables sphinx nose setuptools_scm future h5py
```

* Not all packages are available in the conda repositories, but they can be
easily installed with pip:

```
pip install pyscaffold pytest pytest-cov
```


## And Finally: install wetb

```
git clone https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git
cd WindEnergyToolbox
pip install -e .
```

Note that ```pip install -e .``` will install ```wetb``` in the source directory.
This works best if you are also developing and regularly updating this package.

You can run the tests cases from the source root directory:

```
python setup.py test
```

