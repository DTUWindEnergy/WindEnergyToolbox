'''
Created on 02/11/2015

@author: MMPE
'''

import numpy as np
from wetb.signal.filters import replacer
def replace_by_mean(x):
    return replacer.replace_by_mean(x, np.isnan(x))


def replace_by_line(x):
    return replacer.replace_by_line(x, np.isnan(x))

def replace_by_polynomial(x, deg=3, no_base_points=12):
    return replacer.replace_by_polynomial(x, np.isnan(x), deg, no_base_points)

def max_no_nan(x):
    return replacer.max_cont_mask_length(np.isnan(x))


