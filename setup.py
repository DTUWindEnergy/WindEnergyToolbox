#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wetb.

    This file was generated with PyScaffold 2.5, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import os
import sys
from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext


def setup_package():

    path = 'wetb/fatigue_tools/rainflowcounting/'
    module = 'wetb.fatigue_tools.rainflowcounting'
    names = ['pair_range', 'peak_trough', 'rainflowcount_astm']
    extlist = [Extension('%s.%s' % (module, n),
                         [os.path.join(path, n)+'.pyx']) for n in names]

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          cmdclass = {'build_ext': build_ext},
          ext_modules = extlist,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
