# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:06:06 2018

@author: shfe

Hyperbolic model used for soil load displacement behaviour
"""

import numpy as np

def hyperbolic_load(y, pu, a, b):
    """
    Static hyperbolic model developed by Lee
    
    Parameters
    ----------------
    y -- displacement
    pu -- ultimate load
    a, b -- coefficients
    
    Output
    ----------------
    p -- soil resistence pressure
    """
    
    p = y/(a + b*y) * pu
        
    return p
    
def hyperbolic_disp(p, pu, a, b):
    """
    Static hyperbolic model developed by Lee
    
    Parameters
    ----------------
    p -- load
    pu -- ultimate load
    a, b -- coefficients
    
    Output
    ----------------
    y -- displacement
    """
    
    y = p * a/(pu - p*b)
        
    return y