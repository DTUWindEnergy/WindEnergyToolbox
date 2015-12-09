#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wafo.

    This file was generated with PyScaffold 2.4.2, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

from __future__ import division, absolute_import, print_function

# numpy.distutils will figure out if setuptools is available when imported
# this allows us to combine setuptools use_pyscaffold=True and f2py extensions
import setuptools
from numpy.distutils.core import setup
#from numpy.distutils.misc_util import Configuration

import sys


def setup_package_pyscaffold():

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
          tests_require=['pytest_cov', 'pytest'],
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package_pyscaffold()
