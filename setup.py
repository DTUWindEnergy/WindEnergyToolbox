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

try:
    from pypandoc import convert_file
    read_md = lambda f: convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext


def setup_package():

    ex_info = [('wetb.fatigue_tools.rainflowcounting', ['pair_range', 'peak_trough', 'rainflowcount_astm']),
			   ('wetb.signal.filters', ['cy_filters'])]
    extlist = [Extension('%s.%s' % (module, n),
                         [os.path.join(module.replace(".","/"), n)+'.pyx'],
                         include_dirs=[np.get_include()]) for module, names in ex_info for n in names]

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          cmdclass = {'build_ext': build_ext},
          ext_modules = extlist,
          use_pyscaffold=True,
          long_description=read_md('README.md'))


if __name__ == "__main__":
    setup_package()
