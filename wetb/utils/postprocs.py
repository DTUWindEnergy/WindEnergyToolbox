# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:21:18 2024

@author: nicgo
"""

import numpy as np
import xarray as xr
from wetb.fatigue_tools.fatigue import eq_load
from wetb.utils.envelope import projected_extremes
                                        
def statistics(time, data, info, statistics=['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']):
    def get_stat(stat):
        if hasattr(np, stat):
            return getattr(np, stat)(data, 0)
        elif (stat.startswith("eq") and stat[2:].isdigit()):
            m = float(stat[2:])
            return [eq_load(sensor, 46, m, time[-1] - time[0] + time[1] - time[0])[0][0] for sensor in data.T]
    data = np.array([get_stat(stat) for stat in statistics]).T
    dims = ['sensor_name', 'statistic']
    coords = {'statistic': statistics,
              'sensor_name': info['attribute_names'],
              'sensor_unit': ('sensor_name', info['attribute_units']),
              'sensor_description': ('sensor_name', info['attribute_descriptions'])}
    return xr.DataArray(data=data, dims=dims, coords=coords)


def extreme_loads(data, info, sensors_info, angles=np.linspace(-150,180,12), sweep_angle=30, degrees=True):
    extreme_loads = []
    for sensor, ix, iy in sensors_info:
        extreme_loads.append(projected_extremes(np.vstack([data[:,ix],data[:,iy]]).T, angles, sweep_angle, degrees)[:,1])
    data = np.array(extreme_loads)
    dims = ['sensor_name', 'angle']
    coords = {'angle': angles,
              'sensor_name': [s[0] for s in sensors_info],
              'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in sensors_info]),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)

def remove_time_series(file_h5py):
    time_series_in_file = False
    for group in file_h5py:
        if group.startswith('block'):
            time_series_in_file = True
            del file_h5py[group]
    if time_series_in_file:
        print("Done")
    else:
        print("No time series already")
   
    
def remove_postproc(file_h5py, remove_all=False, postproc_list=[]):
    if remove_all:
        try:
            del file_h5py['postproc']
            print("All postprocs removed'")
        except KeyError:
            print("No postprocs already")
    else:
        for postproc in postproc_list:
            try:
                del file_h5py['postproc'][postproc.__name__]
                print(f"{postproc.__name__} removed'")
            except KeyError:
                print(f"No {postproc.__name__} already'") 
