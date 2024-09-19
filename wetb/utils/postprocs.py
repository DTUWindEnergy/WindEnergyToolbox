# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:21:18 2024

@author: nicgo
"""

import numpy as np
from wetb.fatigue_tools.fatigue import eq_load
from wetb.utils.envelope import projected_extremes

def get_statistics(time, data, statistics=['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']):
    def get_stat(stat):
        if hasattr(np, stat):
            return getattr(np, stat)(data, 0)
        elif (stat.startswith("eq") and stat[2:].isdigit()):
            m = float(stat[2:])
            return [eq_load(sensor, 46, m, time[-1] - time[0] + time[1] - time[0])[0][0] for sensor in data.T]
    return {'Statistics': {'data': np.array([get_stat(stat) for stat in statistics]).T.astype(float), 
                           "statistic_names": np.array([v.encode('utf-8') for v in statistics])},
            'data_array_info': {'dims': ['sensor_name', 'statistic'],
                                'coords': {'statistic': statistics},
                                'coords_function': lambda info, config: {'sensor_name': info['attribute_names'],
                                                                 'sensor_unit': ('sensor_name', info['attribute_units']),
                                                                 'sensor_description': ('sensor_name', info['attribute_descriptions']),
                                                                 }}}


def get_extreme_loads(data, sensors_info, angles=np.linspace(-150,180,12), sweep_angle=30, degrees=True):
    extreme_loads = []
    for sensor, ix, iy in sensors_info:
        extreme_loads.append(projected_extremes(np.vstack([data[:,ix],data[:,iy]]).T, angles, sweep_angle, degrees)[:,1])
    return {'Extreme_loads': {'data': np.array(extreme_loads).astype(float),
                              "sensor_names": np.array([s[0].encode('utf-8') for s in sensors_info]),
                              "angles": angles.astype(float)},
            'data_array_info': {'dims': ['sensor_name', 'angle'],
                                'coords': {'angle': angles},
                                'coords_function': lambda info, config: {'sensor_name': [s[0] for s in config[get_extreme_loads]['sensors_info']],
                                                                         'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in config[get_extreme_loads]['sensors_info']]),
                                                                         }}}
                                           
