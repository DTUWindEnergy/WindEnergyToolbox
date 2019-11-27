# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:15:25 2018

@author: shfe@dtu.dk

Static p-y soil model and cyclic degradation model
"""

import numpy as np

def py_static(y, pu, yc):
    """
    Static p-y model developed by Matlock
    
    Parameters
    ----------------
    y -- displacement
    pu -- ultimate stress
    yc -- ultimate displacement
    
    Output
    ----------------
    p -- soil resistence pressure
    """
    
    if y/yc <= 8:
        p = 0.5*pu*(y/yc)**(1/3)
        
    else:
        p = pu
        
    return p
    
def strain_static(p, pu, yc):
    """
    Static strain using p-y model developed by Matlock
    
    Parameters
    ----------------
    p -- stress
    pu -- ultimate stress
    yc -- ultimate displacement
    
    Output
    ----------------
    y -- soil strain
    """
    if p <= pu:
        y = (p/(0.5*pu))**3*yc
    
    else:
        raise ValueError("The applied stress is LARGER than ultimate strength")
    
    return y
    
def load_unload_py(p, pu, yc):
    """
    load/unload module using p-y model developed by Matlock
    
    Parameters
    ----------------
    p -- stress
    pu -- ultimate stress
    yc -- ultimate displacement
    
    Output
    ----------------
    y -- soil strain
    y_cum -- cumulative soil displacement
    """
    
    y = strain_static(p, pu, yc)
    
    if p <= 0.5*pu:  # elastic
        y_cum = 0
        
    else:
        y_cum = y - p/(0.5*pu/yc)
        
    return y, y_cum



    
def degra_coeff(N, y1, b):
    """
    Degradation factor for p-y model
    
    Parameters
    ----------------
    N -- number of cycle
    y1 -- displacement predicted by static p-y curve
    b -- pile diameter
    
    Output
    ----------------
    lambda -- Degradation factor
    """
    # check

    lambda_N = np.abs(y1/(0.2*b)*np.log(N))
    
    try:
        lambda_N <= 1
    except:
        print("soil must be degraded!!!")
    
    return lambda_N
    
def py_cyclic(y, pu, yc, N, y1, b):
    """
    Degradation p-y model with cyclic load
    
    Parameters
    ----------------
    y -- displacement
    pu -- ultimate stress
    yc -- ultimate displacement
    N -- number of cycle
    y1 -- displacement predicted by static p-y curve
    b -- pile diameter
    
    Output
    ----------------
    p -- degraded soil resistence pressure
    """
    
    lambda_N = degra_coeff(N, y1, b)
    p = py_static(y, pu, yc) * (1 - lambda_N)

    return p

def py_parcel(load_parcel, cycle_parcel, pu, yc, b):
    """
    Degradation p-y model after a parcel of cyclic load
    
    Parameters
    ----------------
    load_parcel -- load parcel
    cycle_parcel -- number of cycle of load parcel
    yc -- ultimate displacement
    b -- pile diameter
    
    Output
    ----------------
    p -- degraded soil resistence pressure
    """
    
    pu_N = np.zeros(np.size(load_parcel)+1)
    pu_N[0] = pu
    lambda_N = np.zeros(np.size(load_parcel))
    y_N = np.zeros(np.size(load_parcel))
    
    for i in np.arange(np.size(load_parcel)):
        load_i = load_parcel[i]
        cycle_i = cycle_parcel[i]
        if np.isnan(load_i):
            y_i = 0
        else:
            y_i = strain_static(load_i, pu_N[i], yc)
        y_N[i] = y_i
        lambda_i = degra_coeff(cycle_i, y_i, b)
        lambda_N[i] = lambda_i
        pu_N[i+1] = pu_N[i] * (1-lambda_i)
    
    return y_N, lambda_N, pu_N
