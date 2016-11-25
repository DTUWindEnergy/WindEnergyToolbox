# Installation

Detailed installation instructions, including how to install Python from scratch,
are described in the [detailed installation manual](docs/install-manual-detailed.md).


If you know what you are doing, you can install as a package as follows:

```
python setup.py install
```

Or in development mode, install from your working directory

```
pip install -e ./
```


Or create a binary wheel distribution package with:

```
python setup.py bdist_wheel -d dist
```



## Works with Python 2 and Python 3

This module is tested for Python 2 and 3 compatibility, and works on both
Windows and Linux. Testing for Mac is on the way, but in theory it should work.
Python 2 and 3 compatibility is achieved with a single code base with the help
of the Python module [future](http://python-future.org/index.html).

Switching to Python 3 is in general a very good idea especially since Python 3.5
was released. Some even dare to say it
[is like eating your vegetables](http://nothingbutsnark.svbtle.com/porting-to-python-3-is-like-eating-your-vegetables).
So if you are still on Python 2, we would recommend you to give Python 3 a try!

You can automatically convert your code from Python 2 to 3 using the
[2to3](https://docs.python.org/2/library/2to3.html) utility which is included
in Python 2.7 by default. You can also write code that is compatible with both
2 and 3 at the same time (you can find additional resources in
[issue 1](https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox/issues/1)).



## Dependencies

* [numpy](http://www.numpy.org/)

* [cython](http://cython.org/)

* [scipy](http://scipy.org/scipylib/)

* [pandas](http://pandas.pydata.org/)

* xlrd and xlwt from [python-excel](http://www.python-excel.org/)

* [openpyxl](http://openpyxl.readthedocs.org/en/default/)

* h5py

* [matplotlib](http://matplotlib.org/)

* [pytables](http://www.pytables.org/)

* [pyscaffold](http://pyscaffold.readthedocs.org/en/)

* pytest, pytest-cov

* six, [future](http://python-future.org/index.html)


## Note

This project has been set up using PyScaffold 2.5. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.

