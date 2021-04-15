.. _installation:

Installation
===========================


Install C++ compilers (Windows)
--------------------------------

Your computer must have a C++ compiler installed in order to install
``wetb``. Windows users can follow the instructions on 
`this page <https://wiki.python.org/moin/WindowsCompilers>`_ under
the section "14.2 standalone".


Normal user
--------------------------------

You must first have installed a C++ compiler before you can use these
commands.

We recommend using the Anaconda Python distribution for easy
installation and maintenance of Python packages.

* Install the most recent, stable version of the code::
  
    pip install wetb

* Update an installation to the most recent version::

    pip install --upgrade wetb

* Install a specific version on PyPI::

   pip install wetb==0.0.21


Advanced user
--------------------------------

Clone the repository and install a local editable copy::

  git clone https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git
  cd WindEnergyToolbox
  pip install -e .

