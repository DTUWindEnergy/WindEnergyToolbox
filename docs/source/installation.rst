.. _installation:

Installation
===========================


Normal user
--------------------------------

* Quick install::

    pip install wetb

* Install a specific version on PyPI::

   pip install wetb==0.0.21

* Update an installation to the most recent version::

    pip install --upgrade wetb
	
* Install with dependencies needed by prepost
  
    pip install wetb[prepost]
	
* Install with all dependencies 
  
    pip install wetb[all]

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
  pip install -e .[all]
  
  
  
  

