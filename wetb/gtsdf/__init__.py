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
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

d = None
d = dir()

from .gtsdf import save
from .gtsdf import load
from .gtsdf import append_block
from .gtsdf import load_pandas

class Dataset(object):
    def __init__(self, filename):
        self.time, self.data, self.info = load(filename)
    def __call__(self, id):
        if isinstance(id, str):
            if id=="Time":
                return self.time
            else:
                return self(self.info['attribute_names'].index(id) + 2)
        if id == 1:
            return self.time
        else:
            return self.data[:, id - 2]
        
    def attribute_names_str(self):
        return "\n".join(["1: Time"]+["%d: %s"%(i,n) for i, n in enumerate(self.info['attribute_names'],2)])
    
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
            
        except Exception as e:
            try:
                return self(name)
            except:
                raise e
            


__all__ = sorted([m for m in set(dir()) - set(d)])




