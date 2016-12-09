# Developer guide

Thank you for your interest in developing wetb. This guide details how to
contribute to wetb in a way that is efficient for everyone.


## Contents

- [Fork](#fork-project)
- [Requirements](#requirements)
- [Install Python](#install-python)
- [Install/build dependencies](#installbuild-dependencies)
- [Get wetb](#get-wetb)
- [Install wetb](#install-wetb)
- [Contributions](#contributions)
- [Upload contributions](#upload-contributions)
- [Make and upload wheels](#make-and-upload-wheels)


## Fork project

We prefer that you make your contributions in your own fork of the project,
[make your changes](#Contributions) and [make a merge request](#Upload contributions).

The project can be forked to your own user account via the \<Fork\> button on
the [frontpage](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox)


## Requirements


### Command line

This guide will use the command line (aka command prompt) frequently.
You can launch a Windows terminal as follows: press Start> and type
"cmd" + \<Enter\>. A link to the command prompt should be visible now.

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

* On windows we highly recommend [tortoisegit](https://tortoisegit.org/). It
is a gui integrated into the windows explorer.


## Install Python
For all platforms we recommend that you download and install the Anaconda -
a professional grade, full blown scientific Python distribution.


### Installing Anaconda, activate root environment

* Download and install Anaconda (Python 3.5 version, 64 bit installer is
recommended) from <https://www.continuum.io/downloads>

> Note: The Python 2.7 or Python 3.5 choice of Anaconda only affects the
root environment. You can always create additional environments using other
Python versions, see below.

* Update the root Anaconda environment (type in a terminal):

```
>> conda update --all
```

* Activate the Anaconda root environment in a terminal as follows:

```
>> activate
```

and your terminal will do something like:
```
C:\Users\> activate
[Anaconda3] C:\Users\>
```
note that the name of the environment is now a prefix before the current path.

use ```deactivate``` to deactivate the environment.


### Optionally, create other independent Anaconda environments

By using environments you can manage different Python installations with
different versions on your system. Creating environments is as easy as:

```
>> conda create -n py27 python=2.7
>> conda create -n py34 python=3.4
>> conda create -n py35 python=3.5
```

These environments can be activated as follows:

```
>> activate py27
>> activate py34
>> activate py35
```

The Python distribution in use will now be located in e.g. \<path_to_anaconda\>/env/py35/

use ```deactivate``` to deactivate the environment.


## Install/build dependencies

- Compiler (```wetb``` contains cython extensions that require a compiler):
    - Linux: gcc (should be installed by default)
    - Windows:
        - Python 2.7: [Microsoft Visual C++ Compiler for Python 2.7](http://aka.ms/vcpython27),
        or the [direct link](https://www.microsoft.com/en-gb/download/details.aspx?id=44266).
        - Python 3.4: MS Visual Studio 2010
        - Python 3.5: MS Visual Studio 2015 or [Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools)
        - Only one MS Visual Studio version can be installed, but you can for
        example install MS Visual Studio 2010 alongside the Visual C++ Build Tools.
- [numpy](http://www.numpy.org/)
- [cython](http://cython.org/)
- [scipy](http://scipy.org/scipylib/)
- [pandas](http://pandas.pydata.org/)
- xlrd and xlwt from [python-excel](http://www.python-excel.org/)
- [openpyxl](http://openpyxl.readthedocs.org/en/default/)
- [h5py](http://www.h5py.org/)
- [matplotlib](http://matplotlib.org/)
- [pytables](http://www.pytables.org/)
- [pyscaffold](http://pyscaffold.readthedocs.org/en/)
- pytest, pytest-cov
- six, [future](http://python-future.org/index.html)
- [parimeko](http://www.paramiko.org/)

Install the necessary Python dependencies using the conda package manager:

```
>> conda install setuptools_scm future h5py pytables pytest pytest-cov nose sphinx blosc pbr paramiko
>> conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl psutil pandoc
>> conda install -c https://conda.anaconda.org/conda-forge pyscaffold --no-deps
```

Note that ```--no-deps``` avoids that newer packages from the channel
```conda-forge``` will be used instead of those from the default ```anaconda```
channel. Depending on which packages get overwritten, this might brake your
Anaconda root environment. As such, using ```--no-deps``` should be
be used for safety (especially when operating from the root environment).


## Get wetb

Copy the https - link on the front page of your fork of wetb

```
>> git clone <https-link>
```

or via tortoise-git:

- Right-click in your working folder
- "Git Clone..."
- \<Ok\>


## Install wetb

```
>> cd WindEnergyToolbox
>> pip install -e . --no-deps
```

Note that the ```no-deps``` option here is used for the same reason as explained
above for the ```conda-forge``` channel: it is to avoid that pip will replace
newer packages compared to the ones as available in the ```Anaconda``` channel.


## Contributions

If you make a change in the toolbox, that others can benefit from please make a merge request.

If you can, please submit a merge request with the fix or improvements including tests.

The workflow to make a merge request is as follows:

- Create a feature branch, branch away from master
- Write tests and code
- Push the commit(s) to your fork
- Submit a merge request (MR) to the master branch of
- Link any relevant issues in the merge request description and leave a comment on them with a link back to the MR
- Your tests should run as fast as possible, and if it uses test files, these files should be as small as possible.
- Please keep the change in a single MR as small as possible. Split the functionality if you can


## Upload contributions

To be written


## Make and upload wheels to PyPi

Uploading wheels to [PyPi](https://pypi.python.org/pypi) is easy thanks to
[```twine```](https://pypi.python.org/pypi/twine).

Install ```twine``` using conda:
```
>> conda install --channel https://conda.anaconda.org/pbrod twine --no-deps
```

Or pip:
```
>> pip install twine
```

One additional complication with PyPi is that the package description is required
to be in the ```rst``` format, while the ```wetb``` readme file is currently
formatted in ```md```. To solve this,
[```pypandoc```](https://pypi.python.org/pypi/pypandoc) can be used to convert
```README.md``` on the fly to ```rst```. A discussion on how this can be done
is also recorded in issue #22.

Install ```pypandoc``` using conda:
```
>> conda install pandoc
```

Or pip:
```
>> pip install pypandoc
```

Workflow for creating and uploading wheels is as follows:

- Make tag: ```git tag "vX.Y.Z"```, and push tag to remote: ```git push --tags```
- In order to have a clean version number (which is determined automagically)
make sure your git working directory is clean (no uncommitted changes etc).
- ```pip install -e . --upgrade```
- ```python setup.py bdist_wheel -d dist``` (wheel includes compiled extensions)
- On Linux you will have to rename the binary wheel file
(see [PEP 513](https://www.python.org/dev/peps/pep-0513/) for a background discussion):
    - from: ```wetb-0.0.5-cp35-cp35m-linux_x86_64.whl```
    - to: ```wetb-0.0.5-cp35-cp35m-manylinux1_x86_64.whl```
- ```python setup.py sdist -d dist``` (for general source distribution installs)
- ```twine upload dist/*```

In case of problems:

- Make sure the version tag is compliant with
[PEP 440](https://www.python.org/dev/peps/pep-0440/), otherwise ```twine upload```
will fail. This means commit hashes can not be part of the version number.

