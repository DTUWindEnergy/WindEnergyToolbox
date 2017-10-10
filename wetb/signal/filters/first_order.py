'''
Created on 10/01/2015

@author: mmpe
'''
import numpy as np

def low_pass(input, delta_t, tau, method=1):
    from wetb.signal.filters import cy_filters

    if isinstance(tau, (int, float)):
        return cy_filters.cy_low_pass_filter(input.astype(np.float64), delta_t, tau)
    else:
        if len(input.shape)==2:
            return cy_filters.cy_dynamic_low_pass_filter_2d(input.astype(np.float64), delta_t, tau, method)
        else:
            return cy_filters.cy_dynamic_low_pass_filter(input.astype(np.float64), delta_t, tau, method)

def high_pass(input, delta_t, tau):
    from wetb.signal.filters import cy_filters

    return cy_filters.cy_high_pass_filter(input.astype(np.float64), delta_t, tau)
