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
commands. See section above.

* Quick install::

    pip install wetb

* Install a specific version on PyPI::

   pip install wetb==0.0.21

* Update an installation to the most recent version::

    pip install --upgrade wetb

**NOTE**. Dependency conflicts can arise if you do not install
``wetb`` into a clean environment. In particular, your installation
might break if ``wetb`` is installed using ``pip``, and then later
packages are installed using ``conda``. (See more details at
`this article <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_.
We therefore recommend that you install ``wetb`` in a clean
environment.


Advanced user
--------------------------------

Clone the repository and install a local editable copy::

  git clone https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git
  cd WindEnergyToolbox
  pip install -e .

