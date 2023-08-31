#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wetb.
"""

from distutils.extension import Extension
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


def setup_package(build_ext_switch=True):
    # if build_ext_switch:
    #     import numpy as np
    #     ex_info = [('wetb.fatigue_tools.rainflowcounting', ['pair_range', 'peak_trough', 'rainflowcount_astm']),
    #                ('wetb.signal.filters', ['cy_filters'])]
    #     extlist = [Extension('%s.%s' % (module, n),
    #                          [os.path.join(module.replace(".", "/"), n) + '.pyx'],
    #                          include_dirs=[np.get_include()]) for module, names in ex_info for n in names]
    #     from Cython.Distutils import build_ext
    #     build_requires = ['cython']
    #     cmd_class = {'build_ext': build_ext}
    # else:
    extlist = []
    build_requires = []
    cmd_class = {}

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    install_requires = ['mock',
                        'h5py',
                        # 'tables', # Has blosc2 as requirement, fails unless C compiler is present on win32
                        'pytest',
                        'pytest-cov',
                        #                        'blosc', # gives an error - has to be pre-installed
                        'pbr',
                        'scipy',
                        'pandas',
                        'matplotlib',
                        'coverage',
                        'psutil',
                        'Click',
                        'jinja2', ]
    extras_require = {
        'prepost': [
            'tables',  # requires blosc2 and may not install on 32 bit systems without a C compiler
            'cython',
            'openpyxl',
        ],
        'all': ['tables',  # requires blosc2 and may not install on 32 bit systems without a C compiler
                'cython',
                'sshtunnel',
                'openpyxl',
                'paramiko',
                ]
    }

    setup(install_requires=install_requires,
          extras_require=extras_require,
          setup_requires=install_requires + build_requires + sphinx,
          cmdclass=cmd_class,
          ext_modules=extlist,
          description=long_description,
          description_content_type="text/markdown",
          version=version,
          packages=find_packages(),
          package_data={'wetb': ['wind/turbulence/mann_spectra_data.npy']},
          )


if __name__ == "__main__":
    try:
        setup_package()
    except Exception:
        setup_package(build_ext_switch=False)
        warnings.warn("WETB installed, but building extensions failed (i.e. it falls back on the slower pure python implementions)",
                      RuntimeWarning)
