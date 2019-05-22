

# Anaconda (Windows/Mac/Linux)

## Installation

Install the necessary Python dependencies using the ```conda``` package manager:

```
>> conda install blosc
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

## Installation and update

```
>> pip install --upgrade wetb
```


# Works with Python 2 and Python 3

This module is tested for Python 2.7 and 3.4+ compatibility, and works on both
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



