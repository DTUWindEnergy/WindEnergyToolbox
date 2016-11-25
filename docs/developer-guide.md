#Developer guide

Thank you for your interest in developing wetb. This guide details how to contribute to wetb in a way that is efficient for everyone.

##Contents

- [Fork](#Fork-project)
- [Requirements](#Requirements)
- [Install Python](#Install-Python)
- [Install dependencies](#Install-dependencies)
- [Get wetb](#Get-wetb)
- [Install wetb](#Install-wetb) 
- [Contributions](#Contributions)
- [Upload contributions](#Upload-contributions)
- [Make and upload wheels](#Make-and-upload-wheels)
 



##Fork project
We prefer that you make your contributions in your own fork of the project, [make your changes](#Contributions) and [make a merge request](#Upload contributions).

The project can be forked to your own user account via the \<Fork\> button on the [frontpage](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox)


##Requirements

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

* On windows we highly recommend [tortoisegit](https://tortoisegit.org/). It is a gui integrated into the windows explorer. 


##Install Python
For all platforms we recommend that you download and install the Anaconda - a professional grade, full blown scientific Python distribution. 

### Installing Anaconda, activate root environment

* Download and install Anaconda (Python 3.5 version, 64 bit installer is recommended) from <https://www.continuum.io/downloads>

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

* By using environments you can manage different Python installations with
different versions on your system. Creating environments is as easy as:

```
>> conda create -n py27 python=2.7
>> conda create -n py34 python=3.4
>> conda create -n py35 python=3.5
```

* These environments can be activated as follows:

```
>> activate py27
>> activate py34
>> activate py35
```

The Python distribution in use will now be located in e.g. \<path_to_anaconda\>/env/py35/

use ```deactivate``` to deactivate the environment.
 

##Install dependencies

Install the necessary Python dependencies using the conda package manager:

```
>> conda install setuptools_scm future h5py pytables pytest nose sphinx
>> conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl psutil
>> conda install -c https://conda.anaconda.org/conda-forge pyscaffold pytest-cov
```

##Get wetb

Copy the https - link on the front page of your fork of wetb

```
>> git clone <https-link>
```

or via tortoise-git:

- Right-click in your working folder
- "Git Clone..."
- \<Ok\>

##Install wetb
```
>> cd WindEnergyToolbox
>> pip install -e . --no-deps
```

##Contributions

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

##Upload contributions
To be written


##Make and upload wheels

```
>> conda install --channel https://conda.anaconda.org/pbrod twine
```

- Make tag
- pip install -e . --upgrade
- python setup.py bdist_wheel -d dist
- twine upload dist\*

