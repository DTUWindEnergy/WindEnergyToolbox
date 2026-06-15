# -*- coding: utf-8 -*-
"""
Author:
    Bjarne S. Kallesoee


Description:
    Reads all HAWC2 output data formats, HAWC2 ascii, HAWC2 binary and FLEX

call ex.:
    # creat data file object, call without extension, but with parth
    file = ReadHawc2("HAWC2ex/test")
    # if called with ReadOnly = 1 as
    file = ReadHawc2("HAWC2ex/test",ReadOnly=1)
    # no channels a stored in memory, otherwise read channels are stored for reuse

    # channels are called by a list
    file([0,2,1,1])  => channels 1,3,2,2
    # if empty all channels are returned
    file()  => all channels as 1,2,3,...
    file.t => time vector

1. version: 19/4-2011
2. version: 5/11-2015 fixed columns to get description right, fixed time vector (mmpe@dtu.dk)

Need to be done:
    * add error handling for allmost every thing

"""
import numpy as np
import os

from wetb import gtsdf
from wetb.prepost import misc
from wetb.hawc2.Hawc2output import view_sensors

import pandas as pd
from collections.abc import Iterable
import warnings

################################################################################
################################################################################
################################################################################
# Read HAWC2 class
################################################################################


class ReadHawc2(view_sensors):
    """
    """
################################################################################
# read *.sel file


    def __init__(self, FileName, ReadOnly=0):
        
        warnings.warn(
            "ReadHawc2 is deprecated. Use Hawc2output instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(FileName, ReadOnly=ReadOnly)
################################################################################
# Read results in binary format
#####################################################################################
# Main read data call, read, save and sort data

    def __call__(
        self,
        ChVec=None,
        htc=None,
        name=None,
        desc=None,
        label=None,
    ):
        result = super().__call__(
            ChVec=[] if ChVec is None else ChVec,
            htc=htc,
            name=name,
            desc=desc,
            label=label,
        )
        return result.values
        

    def __getitem__(self, slc):
        return self()[slc]
        
################################################################################
################################################################################
################################################################################
# write HAWC2 class, to be implemented
################################################################################

if __name__ == '__main__':
    filename = r'C:\Users\ibesi\repos\WindEnergyToolbox\wetb\hawc2\tests\test_files\IEA_15MW_RWT_Monopile.hdf5'
    res_file = ReadHawc2(filename)
    print(type(res_file))
    print(res_file[2])
