"""
General Time Series Data Format - a HDF5 format for time series
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

d = None
d = dir()

from .ascii2bin import ascii2bin, size_from_file

__all__ = [m for m in set(dir()) - set(d)]
