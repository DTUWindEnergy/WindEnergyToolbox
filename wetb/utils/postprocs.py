# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:21:18 2024

@author: nicgo
"""

import numpy as np
import xarray as xr
from wetb.fatigue_tools.fatigue import eq_load, cycle_matrix, shifted_Goodman_diagram
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


def extreme_loads(data, sensors_info) -> xr.DataArray:
    """
    Calculate the extreme load matrix (Fx, Fy, Fz, Mx, My, Mz) for different sensors
    using as criteria driving force or moment along x, y and z directions,
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
    extreme_loads_data = []
    for component, i_fx, i_fy, i_fz, i_mx, i_my, i_mz in sensors_info:
        extreme_loads_data.append([])
        for i in i_fx, i_fy, i_fz, i_mx, i_my, i_mz:
            time_step_max = np.argmax(data[:, i])
            time_step_min = np.argmin(data[:, i])
            extreme_loads_data[-1].append([data[time_step_max, i_fx], data[time_step_max, i_fy], data[time_step_max, i_fz],
                                     data[time_step_max, i_mx], data[time_step_max, i_my], data[time_step_max, i_mz]])
            extreme_loads_data[-1].append([data[time_step_min, i_fx], data[time_step_min, i_fy], data[time_step_min, i_fz],
                                     data[time_step_min, i_mx], data[time_step_min, i_my], data[time_step_min, i_mz]])
        fres = np.sqrt(data[:, i_fx]**2 + data[:, i_fy]**2)
        mres = np.sqrt(data[:, i_mx]**2 + data[:, i_my]**2)
        time_step_max_fres = np.argmax(fres)
        time_step_max_mres = np.argmax(mres)
        extreme_loads_data[-1].append([data[time_step_max_fres, i_fx], data[time_step_max_fres, i_fy], data[time_step_max_fres, i_fz],
                                 data[time_step_max_fres, i_mx], data[time_step_max_fres, i_my], data[time_step_max_fres, i_mz]])
        extreme_loads_data[-1].append([data[time_step_max_mres, i_fx], data[time_step_max_mres, i_fy], data[time_step_max_mres, i_fz],
                                 data[time_step_max_mres, i_mx], data[time_step_max_mres, i_my], data[time_step_max_mres, i_mz]])       
    data = np.array(extreme_loads_data)
    dims = ['sensor_name', 'driver', 'load']
    coords = {'sensor_name': [s[0] for s in sensors_info],
              'driver': ['Fx_max', 'Fx_min', 'Fy_max', 'Fy_min', 'Fz_max', 'Fz_min',
                         'Mx_max', 'Mx_min', 'My_max', 'My_min', 'Mz_max', 'Mz_min',
                         'Fres_max', 'Mres_max'],
              'load': ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
              'sensor_unit': ('load', ['kN', 'kN', 'kN', 'kNm', 'kNm', 'kNm']),
                }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def directional_extreme_loads(data, info, sensors_info, angles=np.linspace(-150,180,12), sweep_angle=30, degrees=True) -> xr.DataArray:
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
    directional_extreme_loads_data = []
    for sensor, ix, iy in sensors_info:
        directional_extreme_loads_data.append(projected_extremes(np.vstack([data[:,ix],data[:,iy]]).T, angles, sweep_angle, degrees)[:,1])
    data = np.array(directional_extreme_loads_data)
    dims = ['sensor_name', 'angle']
    coords = {'angle': angles,
              'sensor_name': [s[0] for s in sensors_info],
              'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in sensors_info]),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def extremes_and_contemporaneous(data, info, sensor_indices) -> xr.DataArray:
    """
    Calculate the maximum and minimum values for different sensors as well as
    the contemporaneous values of the other sensors at such instants.
    
    Parameters
    ----------
    data : array (Ntimesteps x Nsensors_all)
        Array containing the time series of all sensors
    sensor_indices : list
        List containing the indices of the desired output sensors

    Returns
    -------
    xarray.DataArray ((Nsensors*2) x Nsensors)
        DataArray containing the maximum and minimum values for each sensor
        and the contemporaneous values of the other sensors.
    """
    table = []
    for s in sensor_indices:
        time_step_max = np.argmax(data[:, s])
        time_step_min = np.argmin(data[:, s])
        table.append([data[time_step_max, s2] for s2 in sensor_indices])
        table.append([data[time_step_min, s2] for s2 in sensor_indices])
    data = np.array(table)
    dims = ['driver', 'sensor_description']
    sensor_descriptions = [info['attribute_descriptions'][s] for s in sensor_indices]
    coords = {'driver': [s + '_' + e for s in sensor_descriptions for e in ['max', 'min']],
              'sensor_description': sensor_descriptions,
              'sensor_name': ('sensor_description', [info['attribute_names'][s] for s in sensor_indices]),
              'sensor_unit': ('sensor_description', [info['attribute_units'][s] for s in sensor_indices])}
    return xr.DataArray(data=data, dims=dims, coords=coords)


def equivalent_loads(data, time, info, m_list=[3, 4, 6, 8, 10, 12], neq=None, no_bins=46) -> xr.DataArray:
    """
    Calculate fatigue equivalent loads for different sensors
    for different Woehler slopes.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units
    m_list : list (Nwoehlerslopes), optional
        List containing the different woehler slopes. The default is [3, 4, 6, 8, 10, 12].
    neq : int or float, optional
        Number of equivalent load cycles. The default is the time duration in seconds.
    no_bins : int, optional
        Number of bins in rainflow count histogram. The default is 46.

    Returns
    -------
    xarray.DataArray (Nsensors x Nwoehlerslopes)
        DataArray containing fatigue equivalent loads for
        each Woehler slope in m_list.\n
        Dims: sensor_name, m\n
        Coords: sensor_name, sensor_unit, sensor_description, m

    """
    if neq is None:
        neq = time[-1] - time[0] + time[1] - time[0]
    equivalent_loads_data = [eq_load(sensor, no_bins=no_bins, m=m_list, neq=neq)[0] for sensor in data.T]
    data = np.array(equivalent_loads_data)
    dims = ['sensor_name', 'm']
    coords = {'sensor_name': info['attribute_names'],
              'sensor_unit': ('sensor_name', info['attribute_units']),
              'sensor_description': ('sensor_name', info['attribute_descriptions']),
              'm': m_list,
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def damage_equivalent_bending_moments(time,
                                      data,
                                      info,
                                      sensors_info,
                                      method='modified_sweep',
                                      m_list=[3, 4, 6, 8, 10, 12],
                                      neq=None, 
                                      angles=np.linspace(-150, 180, 12),
                                      degrees=True,
                                      mlc_func=None,
                                      **kwargs):
    '''
    Compute damage equivalent loads of bending moments as in Wind Energ. Sci., 11, 1305–1319, 2026:
    'Enhanced approach to match damage-equivalent loads in rotor blade fatigue testing'.

    Parameters
    ----------
    time : array
        Array containing the simulation time from start to end of writting output.
    data : array
        Array containing the time series of all sensors.
    info : dict
        Dictionary that must contain the following entries:
            - attribute_units: list of sensor units.
    sensors_info : dict
        Dictionary containing variables per sensor (section). Its entries must be either lists
        of same length as the number of sensors or arrays whose first dimension length is also
        equal to the number of sensors. It must contain the following entries:
            - 'name': name of each sensor. (n_sensors,)
            - 'index_Mx': index corresponding to Mx in data for each section. (n_sensors,)
            - 'index_My': index corresponding to My in data for each section. (n_sensors,)
            - 'EIx': E * Ix for each section. Needed for 'modified_sweep' and 'strain'. (n_sensors,)
            - 'EIy': E * Iy for each section. Needed for 'modified_sweep' and 'strain'. (n_sensors,)
            - 'rp': distance from elastic center to shell for each section in each direction. Needed for 'strain'. (n_sensors, n_angles)         
    method : str, optional
        Which method to use ('sweep', 'modified_sweep' or 'strain'). The default is 'modified_sweep'.
    m_list : list, optional
        List containing the different woehler slopes. The default is [3, 4, 6, 8, 10, 12].
    neq : int or float, optional
        Equivalent number of load cycles. The default is the time duration in seconds.
    angles : array, optional
        Array containing the directions in which fatigue loads should be computed.
        The default is np.linspace(-150, 180, 12).
    degrees : bool, optional
        Whether angles are in degrees (True) or radians (False). The default is True.
    mlc_func : func, optional
        Mean load correction function to use. The default is None (do not perform mean load correction).
    **kwargs : 
        Additional keyword arguments other than ampls and means needed by mlc_func. Currently only shifted_Goodman_diagram
        is supported, so either epsz_u_t and eps_u_c (n_sensors,) or L_u_t and L_u_c (n_sensors, n_angles) can be passed.

    Returns
    -------
    xarray.DataArray
        DataArray containing equivalent DELs for each section and angle.

    '''
    if neq is None:
        neq = time[-1] - time[0] + time[1] - time[0]
    if degrees:
        angles = np.deg2rad(angles)
    damage_equivalent_bending_moments_data = np.zeros((len(sensors_info['name']), len(angles), len(m_list)))
    for i in range(len(sensors_info['name'])):
        for j in range(len(angles)):
            if method == 'sweep':
                var = np.sin(angles[j]) * data[:, sensors_info['index_Mx'][i]] - np.cos(angles[j]) * data[:, sensors_info['index_My'][i]]
            if method == 'modified_sweep':
                var = np.sin(angles[j]) * data[:, sensors_info['index_Mx'][i]] - np.cos(angles[j]) * sensors_info['EIx'][i] / sensors_info['EIy'][i] * data[:, sensors_info['index_My'][i]]
            if method == 'strain':
                var = sensors_info['rp'][i][j] * np.sin(angles[j]) / sensors_info['EIx'][i] * data[:, sensors_info['index_Mx'][i]] - sensors_info['rp'][i][j] * np.cos(angles[j]) / sensors_info['EIy'][i] * data[:, sensors_info['index_My'][i]]
            
            if mlc_func is None:
                damage_equivalent_bending_moments_data[i, j, :] = eq_load(var, m=m_list, neq=neq)[0]
            if mlc_func == shifted_Goodman_diagram:
                if 'L_u_c' in kwargs and 'L_u_t' in kwargs:
                    if method == 'strain':
                        damage_equivalent_bending_moments_data[i, j, :] = eq_load(var, m=m_list, neq=neq, mlc_func=mlc_func, L_u_t=kwargs['L_u_t'][i][j] * sensors_info['rp'][i][j] / sensors_info['EIx'][i], L_u_c=kwargs['L_u_c'][i][j] * sensors_info['rp'][i][j] / sensors_info['EIx'][i])[0]
                    else:
                        damage_equivalent_bending_moments_data[i, j, :] = eq_load(var, m=m_list, neq=neq, mlc_func=mlc_func, L_u_t=kwargs['L_u_t'][i][j], L_u_c=kwargs['L_u_c'][i][j])[0]
                elif 'epsz_u_c' in kwargs and 'epsz_u_t' in kwargs:
                    if method == 'strain':
                        damage_equivalent_bending_moments_data[i, j, :] = eq_load(var, m=m_list, neq=neq, mlc_func=mlc_func, L_u_t=kwargs['epsz_u_t'][i], L_u_c=kwargs['epsz_u_c'][i])[0]
                    else:
                        damage_equivalent_bending_moments_data[i, j, :] = eq_load(var, m=m_list, neq=neq, mlc_func=mlc_func, L_u_t=sensors_info['EIx'][i] / sensors_info['rp'][i][j] * kwargs['epsz_u_t'][i], L_u_c=sensors_info['EIx'][i] / sensors_info['rp'][i][j] * kwargs['epsz_u_c'][i])[0]
    data = damage_equivalent_bending_moments_data
    dims = ['sensor_name', 'angle', 'm']
    coords = {'sensor_name': sensors_info['name'],
              'angle': np.rad2deg(angles) if degrees else angles,
              'm': m_list,
              'sensor_unit': ('sensor_name', ['-' if method == 'strain' else info['attribute_units'][sensors_info['index_Mx'][i]] for i in range(len(sensors_info['name']))]),
               'method': method}
    return xr.DataArray(data=data, dims=dims, coords=coords)


def directional_equivalent_loads(data, time, info, sensors_info, m_list=[3, 4, 6, 8, 10, 12], neq=None, no_bins=46, angles=np.linspace(-150,180,12), degrees=True) -> xr.DataArray:
    """
    Calculate directional fatigue equivalent loads for different load sensors
    for different Woehler slopes for different directions.

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
    m_list : list (Nwoehlerslopes), optional
        List containing the different woehler slopes. The default is [3, 4, 6, 8, 10, 12].
    neq : int or float, optional
        Number of equivalent load cycles. The default is the time duration in seconds.
    no_bins : int, optional
        Number of bins in rainflow count histogram. The default is 46.
    angles : array (Ndirections), optional
        Array containing the directions along which fatigue loads should be computed.
        The default is np.linspace(-150,180,12).
    degrees : bool, optional
        Whether angles are in degrees (True) or radians (False). The default is True.

    Returns
    -------
    xarray.DataArray (Nsensors x Ndirections x Nwoehlerslopes)
        DataArray containing directional fatigue loads for each sensor in sensors_info, for each Woehler slope in m_list
        and for each direction in angles.\n
        Dims: sensor_name, angle, m\n
        Coords: sensor_name, angle, m, sensor_unit

    """
    if neq is None:
        neq = time[-1] - time[0] + time[1] - time[0]
    directional_equivalent_loads_data = []
    for sensor, ix, iy in sensors_info:
        directional_equivalent_loads_data.append([])
        for angle in angles:
            directional_equivalent_loads_data[-1].append([])
            for m in m_list:
                directional_equivalent_loads_data[-1][-1].append(eq_load(data[:, [ix, iy]] @ projection_2d(angle, degrees=degrees),
                                             no_bins=no_bins,
                                             m=m,
                                             neq=neq)[0][0])
    data = np.array(directional_equivalent_loads_data)
    dims = ['sensor_name', 'angle', 'm']
    coords = {'sensor_name': [s[0] for s in sensors_info],
              'angle': angles,
              'm': m_list,
              'sensor_unit': ('sensor_name', [info['attribute_units'][s[1]] for s in sensors_info]),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def markov_matrices(data, info, no_bins=46) -> xr.DataArray:
    """
    Calculate the Markov matrices for different sensors.

    Parameters
    ----------
    data : array (Ntimesteps x Nsensors)
        Array containing the time series of all sensors
    info : dict
        Dictionary that must contain the following entries:\n
            attribute_names: list of sensor names\n
            attribute_units: list of sensor units
    no_bins : int, optional
        Number of bins in rainflow count histogram. The default is 46.

    Returns
    -------
    xarray.DataArray (Nsensors x Nbins x 2)
        DataArray containing number of cycles and load amplitude 
        for each sensor.\n
        Dims: sensor_name, bin, (cycles, amplitude)\n
        Coords: sensor_name, sensor_unit, sensor_description

    """
    cycles_and_amplitudes = []
    for sensor in data.T:
        try:
            cycles, ampl_bin_mean, _, _, _ = cycle_matrix(sensor,
                                                          ampl_bins=no_bins,
                                                          mean_bins=1)
            cycles_and_amplitudes.append([[cycles.flatten()[i],
                                         ampl_bin_mean.flatten()[i]]
                                         for i in range(no_bins)])
        except TypeError:
            cycles_and_amplitudes.append([[np.nan, np.nan]]*no_bins)
    data = np.array(cycles_and_amplitudes)
    dims = ['sensor_name', 'bin', '(cycles, amplitude)']
    coords = {'sensor_name': info['attribute_names'],
              'sensor_unit': ('sensor_name', info['attribute_units']),
              'sensor_description': ('sensor_name', info['attribute_descriptions']),
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)


def directional_markov_matrices(data, info, sensors_info, no_bins=46, angles=np.linspace(-150,180,12), degrees=True) -> xr.DataArray:
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
