#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for wetb.

Non-dev installation from GitLab: 
  pip install git+https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git

Non-dev installation from local clone:
  pip install .

Dev installation from local clone:
  pip install -e .
"""
from setuptools import setup

setup(name='Wind Energy Toolbox', 
      version='2.0a',  
      description='Wind Energy Toolbox - Common tools for wind energy applications',
      url='https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox',
      author='DTU Wind Energy',  
      author_email='dave@dtu.dk',
      license='MIT',
      packages=['wetb'
                ],
      install_requires=[
        'numpy',  # numerical calculations
        'pytest',  # testing
        'pytest-cov',  # calculating coverage
        'sphinx',  # generating documentation
        'sphinx_rtd_theme'  # docs theme
      ],
      zip_safe=True)
