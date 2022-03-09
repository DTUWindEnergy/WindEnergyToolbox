#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wetb.
"""

import os
from git_utils import write_vers
import sys
from setuptools import setup, find_packages
import warnings

repo = os.path.dirname(__file__)
try:
    version = write_vers(vers_file='wetb/__init__.py', repo=repo, skip_chars=1)
except Warning:
    # when there is not git repo, take version string form wetb/__init__.py
    import wetb
    version = wetb.__version__

# try:
#    from pypandoc import convert_file
#    read_md = lambda f: convert_file(f, 'rst', format='md')
# except ImportError:
#    print("warning: pypandoc module not found, could not convert Markdown to RST")
#    read_md = lambda f: open(f, 'r').read()
with open("README.md", "r") as fh:
    long_description = fh.read()


from distutils.extension import Extension


def setup_package(build_ext_switch=True):
    if build_ext_switch:
        import numpy as np
        ex_info = [('wetb.fatigue_tools.rainflowcounting', ['pair_range', 'peak_trough', 'rainflowcount_astm']),
                   ('wetb.signal.filters', ['cy_filters'])]
        extlist = [Extension('%s.%s' % (module, n),
                             [os.path.join(module.replace(".", "/"), n) + '.pyx'],
                             include_dirs=[np.get_include()]) for module, names in ex_info for n in names]
        from Cython.Distutils import build_ext
        build_requires = ['cython']
        cmd_class = {'build_ext': build_ext}
    else:
        extlist = []
        build_requires = []
        cmd_class = {}

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    install_requires = ['mock',
                        'h5py',
                        'tables',
                        'pytest',
                        'pytest-cov',
                        #                        'blosc', # gives an error - has to be pre-installed
                        'pbr',
                        'paramiko',
                        'scipy',
                        'pandas',
                        'matplotlib',
                        'cython',
                        'coverage',
                        'xlwt',
                        'openpyxl',
                        'psutil',
                        'six',
                        'sshtunnel',
                        'Click',
                        'jinja2', ]

    setup(install_requires=install_requires,
          setup_requires=install_requires + build_requires + sphinx,
          cmdclass=cmd_class,
          ext_modules=extlist,
          long_description=long_description,
          long_description_content_type="text/markdown",
          version=version,
          packages=find_packages(),
          )


if __name__ == "__main__":
    try:
        setup_package()
    except Exception:
        setup_package(build_ext_switch=False)
        warnings.warn("WETB installed, but building extensions failed (i.e. it falls back on the slower pure python implementions)",
                      RuntimeWarning)
