"""
General Time Series Data Format - a HDF5 format for time series
"""

d = None
d = dir()

from .ascii2bin import ascii2bin, size_from_file

__all__ = [m for m in set(dir()) - set(d)]
