# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:21:18 2024

@author: nicgo
"""

import numpy as np
import xarray as xr
from wetb.fatigue_tools.fatigue import eq_load, cycle_matrix
from wetb.utils.envelope import projected_extremes
from wetb.utils.rotation import projection_2d
                                        
def statistics(time, data, info,
               statistics=['min', 'mean', 'max', 'std']) -> xr.DataArray:
    """
    Calculate statistics from different sensors.

    Parameters
    ----------
    time : array (Ntimesteps)
        Array containing the simulation time from start to end of writting output
    data : array (Ntimesteps x Nsensors)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units\n
            attribute_descriptions: list of sensor descriptions
    statistics : list (Nstatistics), optional
        List containing the different types of statistics to calculate for each sensor.
        The default is ['min', 'mean', 'max', 'std'].

    Returns
    -------
    xarray.DataArray (Nsensors x Nstatistics)
        DataArray containing statistics for all sensors.\n
        Dims: sensor_name, statistic\n
        Coords: sensor_name, statistic, sensor_unit, sensor_description

    """
    def get_stat(stat):
        if hasattr(np, stat):
            return getattr(np, stat)(data, 0)
    data = np.array([get_stat(stat) for stat in statistics]).T
    dims = ['sensor_name', 'statistic']
    coords = {'statistic': statistics,
              'sensor_name': info['attribute_names'],
              'sensor_unit': ('sensor_name', info['attribute_units']),
              'sensor_description': ('sensor_name', info['attribute_descriptions'])}
    return xr.DataArray(data=data, dims=dims, coords=coords)


def extreme_loads(data, info, sensors_info, angles=np.linspace(-150,180,12), sweep_angle=30, degrees=True) -> xr.DataArray:
    """
    Calculate extreme loads from different sensors along different directions.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors_all)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units
    sensors_info : list of tuples (Nsensors)
        List of 3-element tuples containing the name of a load sensor and the indices of 2 of their components.\n
        Example: [('Tower base shear force', 0, 1), ('Blade 1 root bending moment', 9, 10)]
    angles : array (Ndirections), optional
        Array containing the directions along which extreme loads should be computed.
        The default is np.linspace(-150,180,12).
    sweep_angle : float, optional
        Angle to be swept to both sides of a direction in the search for its extreme load.
        It should match the angular step in angles. The default is 30.
    degrees : bool, optional
        Whether angles and sweep_angle are in degrees (True) or radians (False). The default is True.

    Returns
    -------
    xarray.DataArray (Nsensors x Ndirections)
        DataArray containing extreme loads for each sensor in sensors_info in each direction in angles.\n
        Dims: sensor_name, angle\n
        Coords: sensor_name, angle, sensor_unit

    """
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


def extreme_loads_matrix(data, sensors_info) -> xr.DataArray:
    """
    Calculate the extreme load matrix (Fx, Fy, Fz, Mx, My, Mz) for different sensors using as criteria driving force or moment along x, y and z directions,
    including also maximum x-y in-plane loads Fres and Mres.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors_all)
        Array containing the time series of all sensors
    sensors_info : list of tuples (Nsensors)
        List of 7-element tuples containing the name of a load sensor and the indices of the 6 of their components.\n
        Example: [('Tower base', 0, 1, 2, 3, 4, 5), ('Blade 1 root', 6, 7, 8, 9, 10, 11)]

    Returns
    -------
    xarray.DataArray (Nsensors x 14)
        DataArray containing extreme load states (Fx, Fy, Fz, Mx, My, Mz) for each sensor in sensors_info
        for each case where a load is maximum or minimum (6 x 2 = 12) plus the 2 cases where Fres and Mres are maximum.\n
        Dims: sensor_name, driver, load\n
        Coords: sensor_name, driver, load, sensor_unit

    """
    extreme_loads = []
    for component, i_fx, i_fy, i_fz, i_mx, i_my, i_mz in sensors_info:
        extreme_loads.append([])
        for i in i_fx, i_fy, i_fz, i_mx, i_my, i_mz:
            time_step_max = np.argmax(data[:, i])
            time_step_min = np.argmin(data[:, i])
            extreme_loads[-1].append([data[time_step_max, i_fx], data[time_step_max, i_fy], data[time_step_max, i_fz],
                                     data[time_step_max, i_mx], data[time_step_max, i_my], data[time_step_max, i_mz]])
            extreme_loads[-1].append([data[time_step_min, i_fx], data[time_step_min, i_fy], data[time_step_min, i_fz],
                                     data[time_step_min, i_mx], data[time_step_min, i_my], data[time_step_min, i_mz]])
        fres = np.sqrt(data[:, i_fx]**2 + data[:, i_fy]**2)
        mres = np.sqrt(data[:, i_mx]**2 + data[:, i_my]**2)
        time_step_max_fres = np.argmax(fres)
        time_step_max_mres = np.argmax(mres)
        extreme_loads[-1].append([data[time_step_max_fres, i_fx], data[time_step_max_fres, i_fy], data[time_step_max_fres, i_fz],
                                 data[time_step_max_fres, i_mx], data[time_step_max_fres, i_my], data[time_step_max_fres, i_mz]])
        extreme_loads[-1].append([data[time_step_max_mres, i_fx], data[time_step_max_mres, i_fy], data[time_step_max_mres, i_fz],
                                 data[time_step_max_mres, i_mx], data[time_step_max_mres, i_my], data[time_step_max_mres, i_mz]])       
    data = np.array(extreme_loads)
    dims = ['sensor_name', 'driver', 'load']
    coords = {'sensor_name': [s[0] for s in sensors_info],
              'driver': ['Fx_max', 'Fx_min', 'Fy_max', 'Fy_min', 'Fz_max', 'Fz_min',
                         'Mx_max', 'Mx_min', 'My_max', 'My_min', 'Mz_max', 'Mz_min',
                         'Fres_max', 'Mres_max'],
              'load': ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
              'sensor_unit': ('load', ['kN', 'kN', 'kN', 'kNm', 'kNm', 'kNm']),
                }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def fatigue_loads(data, info, sensors_info, m_list, neq=1e7, no_bins=46, angles=np.linspace(-150,180,12), degrees=True) -> xr.DataArray:
    """
    Calculate directional fatigue equivalent loads for different load sensors for different Woehler slopes for different directions.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors_all)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units
    sensors_info : list of tuples (Nsensors)
        List of 3-element tuples containing the name of a load sensor and the indices of 2 of their components.\n
        Example: [('Tower base shear force', 0, 1), ('Blade 1 root bending moment', 9, 10)]
    m_list : list (Nwoehlerslopes)
        List containing the different woehler slopes
    neq : int or float, optional
        Number of equivalent load cycles. The default is 1e7.
    no_bins : int, optional
        Number of bins in rainflow count histogram. The default is 46.
    angles : array (Ndirections), optional
        Array containing the directions along which fatigue loads should be computed.
        The default is np.linspace(-150,180,12).
    degrees : bool, optional
        Whether angles are in degrees (True) or radians (False). The default is True.

    Returns
    -------
    xarray.DataArray (Nsensors x Nwoehlerslopes x Ndirections)
        DataArray containing directional fatigue loads for each sensor in sensors_info, for each Woehler slope in m_list
        and for each direction in angles.\n
        Dims: sensor_name, m, angle\n
        Coords: sensor_name, m, angle, sensor_unit

    """
    fatigue_loads = []
    for sensor, ix, iy in sensors_info:
        fatigue_loads.append([])
        for m in m_list:
            fatigue_loads[-1].append([])
            for angle in angles:
                fatigue_loads[-1][-1].append(eq_load(data[:, [ix, iy]] @ projection_2d(angle, degrees=degrees),
                                             no_bins=no_bins,
                                             m=m,
                                             neq=neq)[0][0])
    data = np.array(fatigue_loads)
    dims = ['sensor_name', 'm', 'angle']
    coords = {'sensor_name': [s[0] for s in sensors_info],
              'm': m_list,
              'angle': angles,
              'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in sensors_info]),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def markov_matrices(data, info, sensors_info, no_bins=46, angles=np.linspace(-150,180,12), degrees=True) -> xr.DataArray:
    """
    Calculate the Markov matrices for different load sensors for different directions.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors_all)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units
    sensors_info : list of tuples (Nsensors)
        List of 3-element tuples containing the name of a load sensor and the indices of 2 of their components.\n
        Example: [('Tower base shear force', 0, 1), ('Blade 1 root bending moment', 9, 10)]
    no_bins : int, optional
        Number of bins in rainflow count histogram. The default is 46.
    angles : array (Ndirections), optional
        Array containing the directions along which fatigue loads should be computed.
        The default is np.linspace(-150,180,12).
    degrees : bool, optional
        Whether angles are in degrees (True) or radians (False). The default is True.

    Returns
    -------
    xarray.DataArray (Nsensors x Ndirections x Nbins x 2)
        DataArray containing directional number of cycles and load amplitude 
        for each sensor in sensors_info, for each direction in angles and
        for each bin in no_bins.\n
        Dims: sensor_name, angle, bin, (cycles, amplitude)\n
        Coords: sensor_name, angle, sensor_unit

    """
    cycles_and_amplitudes = []
    for sensor, ix, iy in sensors_info:
        cycles_and_amplitudes.append([])
        for angle in angles:
            cycles, ampl_bin_mean, _, _, _ = cycle_matrix(data[:, [ix, iy]] @ projection_2d(angle, degrees=degrees),
                                         ampl_bins=no_bins,
                                         mean_bins=1)
            cycles_and_amplitudes[-1].append([[cycles.flatten()[i],
                                             ampl_bin_mean.flatten()[i]]
                                             for i in range(no_bins)])
    data = np.array(cycles_and_amplitudes)
    dims = ['sensor_name', 'angle', 'bin', '(cycles, amplitude)']
    coords = {'sensor_name': [s[0] for s in sensors_info],
              'angle': angles,
              'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in sensors_info]),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)

    
def remove_postproc(file_h5py, remove_all=False, postproc_list=[]) -> None:
    """
    Remove postproc data from hdf5 file.

    Parameters
    ----------
    file_h5py : h5py.File object
        h5py.File object in append mode from hdf5 file
    remove_all : bool, optional
        Whether all postprocs should be removed from hdf5 file. The default is False.
    postproc_list : list of functions, optional
        List containing which postproc functions to remove their data from hdf5 file. The default is [].

    Returns
    -------
    None.

    """
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
