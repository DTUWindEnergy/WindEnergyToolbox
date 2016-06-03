
> !! This guide is not finished, and might contain innacuracies. Please report
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

* Basic dependencies:

> python (3.5 recommended) git gcc gcc-fortran (gfortran)

* Use either your system package manager, pip + virtualenv, or Anaconda to
install the following python dependencies:

> numpy, cython, scipy, pandas, matplotlib, pyscaffold, future, nose, sphinx,
> xlrd, (py)tables, h5py, pytest, pytest-cov, setuptools_scm, setuptools

Note that often the pytables packages is called python-tables instead of
python-pytables.


## Dependencies on Mac

People who now how to handle Python on the Mac side are kindly requested to
complete this guide :-)


## Dependencies on Windows

A Python installation with compilers is required. If you already have this,
or know how set up such an environment, you skip to
[here](install-manual-detailed.md#and-finally-install-wetb).


### Microsft Visual Studio 2010 Compiler

```wetb``` contains extensions that need to be compiled.
On Windows things are complicated because you need to use the same compiler as
the one used for Python. This means that for compiling extensions on:
* Python 2.7 you need [Microsoft Visual C++ Compiler for Python 2.7](http://aka.ms/vcpython27),
or the [direct link](https://www.microsoft.com/en-gb/download/details.aspx?id=44266).
* Python 3.4 you need MS Visual Studio 2010
* Python 3.5 (and later) you need MS Visual Studio 2015
* You can install Microsoft Visual C++ Compiler for Python 2.7 alongside
MS Visual Studio 2010, but you can not install Visual Studio 2010 and 2015
in parallel.

You can find more background information and installation instructions
[here](https://packaging.python.org/en/latest/extensions/#setting-up-a-build-environment-on-windows),
[here](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/),
or [here](http://stevedower.id.au/blog/building-for-python-3-5-part-two/).


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


## Recommended python distribution: Anaconda

### Installing Anaconda, activate root environment

* Anaconda is a professional grade, full blown scientific Python distribution.

* Update the root Anaconda environment (type in a terminal):

```
conda update --all
```

* Activate the Anaconda root environment in a terminal as follows:

```
activate
```

and your terminal will do something like:
```
C:\Users\> activate
[Anaconda3] C:\Users\>
```
note that the name of the environment is now a prefix before the current path.

use ```deactivate``` to deactivate the environment.


### Optionally, create other independent Anaconda environments

* By using environments you can manage different Python installations with
different versions on your system. Creating environments is as easy as:

```
conda create -n py27 python=2.7
conda create -n py34 python=3.4
conda create -n py35 python=3.5
```

* These environments can be activated as follows:

```
activate py27
activate py34
activate py35
```

use ```deactivate``` to deactivate the environment.


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
git clone https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git
cd WindEnergyToolbox
pip install -e . --no-deps
```

Note that ```pip install -e . --no-deps``` will install ```wetb``` in the source
directory. This works best if you are also developing and regularly updating
this package.

You can run the tests cases from the source root directory:

```
python setup.py test
```

