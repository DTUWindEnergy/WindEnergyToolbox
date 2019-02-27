#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wetb.
"""

import os
import sys
from setuptools import setup, find_packages
import wetb
__version__ = wetb.__version__

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
                         [os.path.join(module.replace(".", "/"), n) + '.pyx'],
                         include_dirs=[np.get_include()]) for module, names in ex_info for n in names]
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    build_requires = ['setuptools_scm',
                      'future',
                      'h5py',
                      'pytables',
                      'pytest',
                      'pytest-cov',
                      'nose',
                      'blosc',
                      'pbr',
                      'paramiko',
                      'scipy',
                      'pandas',
                      'matplotlib',
                      'cython',
                      'xlrd',
                      'coverage',
                      'xlwt',
                      'openpyxl',
                      'psutil',
                      'six'] + sphinx
    setup(install_requires=build_requires,
          setup_requires=build_requires,
          cmdclass={'build_ext': build_ext},
          ext_modules=extlist,
          long_description=read_md('README.md'),
          version=__version__,
          packages=find_packages(),
          )


if __name__ == "__main__":
    setup_package()
