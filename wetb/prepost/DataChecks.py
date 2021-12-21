# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:00:02 2012

@author: dave
"""

# time and data should be 1 dimensional arrays
def array_1d(array):
    """
    Check if the given array has only one dimension. Following formats will
    return True:
        (x,), (x,1) and (1,x)
    """
    if not len(array.shape) == 1:
        # in case it has (samples,1) or (1,samples) as dimensions
        if len(array.shape) == 2:
            if (array.shape[0] == 1) or (array.shape[1] == 1):
                return True
            else:
                raise ValueError('only 1D arrays are accepted')
    else:
        return True

    return True
