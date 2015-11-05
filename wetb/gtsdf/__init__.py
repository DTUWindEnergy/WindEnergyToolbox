"""
The 'General Time Series Data Format', gtsdf, is a binary hdf5 data format for storing time series data,\n
specified by \n
Mads M. Pedersen (mmpe@dtu.dk), DTU-Wind Energy, Aeroelastic design (AED)

Features:

-    Single file
-    Optional data type, e.g. 16bit integer (compact) or 64 bit floating point (high precision)
-    Precise time representation (including absolute times)
-    Additional data blocks can be appended continuously
-    Optional specification of name and description of dataset
-    Optional specification of name, unit and description of attributes
-    NaN support

This module contains three methods:

- load_
- save_
- append_block_

.. _load: gtsdf.html#gtsdf.load
.. _save: gtsdf.html#gtsdf.save
.. _append_block: gtsdf.html#gtsdf.append_block

"""

d = None
d = dir()

from .gtsdf import save
from .gtsdf import load
from .gtsdf import append_block
from .gtsdf import load_pandas

__all__ = sorted([m for m in set(dir()) - set(d)])




