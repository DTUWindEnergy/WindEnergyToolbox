import numpy as np
import os
import re
import xarray as xr
from copy import copy
from wetb.fatigue_tools.fatigue import eq_loads_from_Markov

def get_DLC(filename):
    filename = os.path.basename(filename)
    i1 = filename.find('DLC')
    i2 = filename.find('_', i1)
    if i2 == -1:
        DLC = filename[i1:]
    else:
        DLC = filename[i1:i2]
    return DLC

def apply_regex(filename, regex):
    filename = os.path.basename(filename)
    if re.match(regex, filename) is None:
        return False
    else:
        return True
    
def get_groups(filename, regex_list):
    for regex in regex_list.values():
        match = re.match(regex, os.path.basename(filename))
        if match is not None:
            break
    return match.group(0) if match is not None else None

def group_simulations(dataarray, regex_list):
    groups = np.unique(np.vectorize(get_groups)(dataarray.filename, regex_list))
    grouped_simulations = {}
    for group in groups:
        grouped_simulations[group] = dataarray.sel(filename=np.vectorize(apply_regex)(dataarray.filename, group))
    return grouped_simulations

def average_contemporaneous_loads(extreme_loads, driver, load, metric, safety_factor):
    if load in driver:  
        return metric(extreme_loads)*safety_factor
    if driver == 'Fres_max':
        if load in ['Fx', 'Fy']:
            return metric(extreme_loads)*safety_factor
    if driver == 'Mres_max':
        if load in ['Mx', 'My']:
            return metric(extreme_loads)*safety_factor
    return np.mean(np.abs(extreme_loads))*np.sign(extreme_loads[np.argmax(np.abs(extreme_loads))])
    
def scale_contemporaneous_loads(extreme_loads, driver, load, scaling_factor, safety_factor):
    if load in driver:  
        return extreme_loads*scaling_factor*safety_factor
    if driver == 'Fres_max':
        if load in ['Fx', 'Fy']:
            return extreme_loads*scaling_factor*safety_factor
    if driver == 'Mres_max':
        if load in ['Mx', 'My']:
            return extreme_loads*scaling_factor*safety_factor
    return extreme_loads*scaling_factor

def get_loads_by_group(extreme_loads, regex_list, metric_list, safety_factor_list, contemporaneous_method='averaging'):   
    grouped_simulations = group_simulations(extreme_loads, regex_list)
    loads_by_group_dict = {}
    if contemporaneous_method == 'averaging':
        for group, simulations in grouped_simulations.items():
            DLC = get_DLC(group)
            group_loads = xr.apply_ufunc(average_contemporaneous_loads,
                                         simulations,
                                         simulations.coords['driver'], 
                                         simulations.coords['load'], 
                                         kwargs={'metric': metric_list[DLC], 'safety_factor': safety_factor_list[DLC]},
                                         input_core_dims=[['filename'], [], []],
                                         vectorize=True)
            Fres = np.sqrt(group_loads.sel(load='Fx')**2 + group_loads.sel(load='Fy')**2)
            Fres.coords['load'] = 'Fres'
            Mres = np.sqrt(group_loads.sel(load='Mx')**2 + group_loads.sel(load='My')**2)
            Mres.coords['load'] = 'Mres'
            thetaF = np.arctan2(group_loads.sel(load='Fy'), group_loads.sel(load='Fx'))*180/np.pi
            thetaF.coords['load'] = 'Theta_F'
            thetaM = np.arctan2(group_loads.sel(load='My'), group_loads.sel(load='Mx'))*180/np.pi
            thetaM.coords['load'] = 'Theta_M'
            group_loads = xr.concat([group_loads, Fres, thetaF, Mres, thetaM], dim='load')
            loads_by_group_dict[group] = group_loads
    elif contemporaneous_method == 'scaling':
        for group, simulations in grouped_simulations.items():
            main_loads = simulations.isel(driver=range(12), load=xr.DataArray([int(i/2) for i in range(12)], dims='driver'))
            Fres = np.sqrt(simulations.sel(driver='Fres_max').sel(load='Fx')**2 + simulations.sel(driver='Fres_max').sel(load='Fy')**2)
            Mres = np.sqrt(simulations.sel(driver='Mres_max').sel(load='Mx')**2 + simulations.sel(driver='Mres_max').sel(load='My')**2)
            Fres.coords['load'] = 'Fres'
            Mres.coords['load'] = 'Mres'
            main_loads = xr.concat([main_loads, Fres, Mres], dim='driver')
            DLC = get_DLC(group)
            characteristic_loads = xr.apply_ufunc(metric_list[DLC],
                                    main_loads,
                                    input_core_dims=[['filename']],
                                    vectorize=True)
            scaling_factors = characteristic_loads/main_loads
            closest_load_indices = np.abs(1/scaling_factors - 1).argmin('filename')
            scaling_factors = scaling_factors.isel(filename=closest_load_indices)
            closest_load_indices = closest_load_indices.drop_vars('load')
            simulations = simulations.isel(filename=closest_load_indices)
            group_loads = xr.apply_ufunc(scale_contemporaneous_loads,
                                         simulations,
                                         simulations.coords['driver'], 
                                         simulations.coords['load'],
                                         scaling_factors,
                                         kwargs={'safety_factor': safety_factor_list[DLC]},
                                         input_core_dims=[[], [], [], []],
                                         vectorize=True)
            Fres = np.sqrt(group_loads.sel(load='Fx')**2 + group_loads.sel(load='Fy')**2)
            Fres.coords['load'] = 'Fres'
            Mres = np.sqrt(group_loads.sel(load='Mx')**2 + group_loads.sel(load='My')**2)
            Mres.coords['load'] = 'Mres'
            thetaF = np.arctan2(group_loads.sel(load='Fy'), group_loads.sel(load='Fx'))*180/np.pi
            thetaF.coords['load'] = 'Theta_F'
            thetaM = np.arctan2(group_loads.sel(load='My'), group_loads.sel(load='Mx'))*180/np.pi
            thetaM.coords['load'] = 'Theta_M'
            group_loads = xr.concat([group_loads, Fres, thetaF, Mres, thetaM], dim='load')
            loads_by_group_dict[group] = group_loads
        
    loads_by_group = xr.concat(list(loads_by_group_dict.values()), 'group')
    if 'variable' in loads_by_group.coords:
        loads_by_group = loads_by_group.drop_vars('variable')
    if 'filename' in loads_by_group.coords:
        loads_by_group = loads_by_group.drop_vars('filename')
    loads_by_group.coords['group'] = list(loads_by_group_dict.keys())
    return loads_by_group

def get_DLB_extreme_loads(extreme_loads, regex_list, metric_list, safety_factor_list, contemporaneous_method='averaging'):
    """
    Calculate the extreme loads for the whole DLB.

    Parameters
    ----------
    dataarray : xarray.DataArray (Nsimulations x Nsensors x 14 x 10)
        DataArray containing collected data for each simulation and load sensor.
        The 14 x 6 matrix corresponds to the extreme loading matrix in IE61400-1
        in annex I-1, where the 10 components are Fx, Fy, Fz, Mx, My, Mz, Fr, theta_F, Mr, theta_M.
        Dims: filename, sensor_name, driver, load
    regex_list : dict
        Dictionary containing the regular expression for grouping simulations
        for each DLC.
    metric_list : dict
        Dictionary containing the function to be applied to each group of simulations
        for each DLC.
    safety_factor_list : dict
        Dictionary containing the safety factor to be applied to each group of simulations
        for each DLC.
    contemporaneous_method : str
        Method for assessing the contemporaneous loads.
        'averaging': the mean of the absolute values from each timeseries is used,
         posteriorly applying the sign of the absolute maximum value.
         'scaling': the contemporaneous values from the timeseries with the closest
         load to the characteristic load are selected, posteriorly scaling them
         by the factor characteristic load / timeseries load.
         The default is 'averaging'.

    Returns
    -------
    DataArray (Nsensors x 14 x 10)
        DataArray containing the extreme loading matrix of the DLB for each sensor,
        as well as the group of simulations driving each sensor.\n
        Dims: sensor_name, driver, load
        Coords: sensor_name, driver, load, group

    """
    loads_by_group = get_loads_by_group(extreme_loads, regex_list, metric_list,
                                        safety_factor_list, contemporaneous_method=contemporaneous_method)
    driving_group_indices = []
    for driver in loads_by_group.coords['driver'].values:
        load = driver[:driver.find('_')]
        metric = driver[driver.find('_') + 1:]
        if metric == 'max':
            driving_group_indices.append(loads_by_group.sel(driver=driver, load=load).argmax('group'))
        elif metric == 'min':
            driving_group_indices.append(loads_by_group.sel(driver=driver, load=load).argmin('group'))
    driving_group_indices = xr.concat(driving_group_indices, dim='driver')
    driving_group_indices = driving_group_indices.drop_vars('load')
    DLB_extreme_loads = loads_by_group.isel(group=driving_group_indices)
    return DLB_extreme_loads

def get_group_values(dataarray, regex_list, metric_list, safety_factor_list):
    grouped_simulations = group_simulations(dataarray, regex_list)
    group_values_dict = {}
    for group, simulations in grouped_simulations.items():
        DLC = get_DLC(group)
        values = xr.apply_ufunc(
        metric_list[DLC], 
        simulations,
        input_core_dims=[["filename"]],
        vectorize=True)*safety_factor_list[DLC]
        group_values_dict[group] = values
    group_values = xr.concat(list(group_values_dict.values()), 'group')
    group_values = group_values.drop_vars('variable')
    group_values.coords['group'] = list(group_values_dict.keys())
    return group_values

def get_extreme_values(dataarray):
    max_values = dataarray.max('group')
    max_indices = dataarray.argmax('group')
    max_values.coords['group'] = dataarray['group'].isel(group=max_indices)
    return max_values            

def get_DLB_extreme_values(dataarray, regex_list, metric_list, safety_factor_list):
    """
    Calculate the extreme values of any sensor for the whole DLB.

    Parameters
    ----------
    dataarray : xarray.DataArray (Nsimulations x *)
        DataArray containing collected data for each simulation. Must have
        filename as leading dimension and can have any number of extra dimensions.
        Dims: filename, *\n
        Coords: filename, *
    regex_list : dict
        Dictionary containing the regular expression for grouping simulations
        for each DLC.
    metric_list : dict
        Dictionary containing the function to be applied to each group of simulations
        for each DLC.
    safety_factor_list : dict
        Dictionary containing the safety factor to be applied to each group of simulations
        for each DLC.

    Returns
    -------
    DataArray (Nsensors x Ndirections x 3)
        DataArray containing the extreme values of the DLB, as well as 
        the group of simulations driving each sensor.\n
        Dims: dataarray.dims without filename\n
        Coords: dataarray.dims without filename, group

    """
    group_values = get_group_values(dataarray, regex_list, metric_list, safety_factor_list)
    extreme_values = get_extreme_values(group_values)
    return extreme_values

def get_DLB_eq_loads(eq_loads, weight_list, neq=1e7):
    """
    Calculate the fatigue equivalent loads for the whole DLB
    from equivalent loads of individual simulations.

    Parameters
    ----------
    eq_loads : xarray.DataArray (Nsimulations x * x Nwoehlerslopes)
        DataArray containing equivalent loads for each Woehler slope, 
        for each sensor and for each simulation.\n
        Dims: filename, *, m\n
        Coords: filename, *, m
    weight_list : dict or xarray.DataArray
        Dictionary or DataArray containing the weight (real time / simulation time) for each simulation
    neq : int or float, optional
        Number of equivalent load cycles. The default is 1e7.

    Returns
    -------
    xarray.DataArray (Nsensors x Ndirections x Nwoehlerslopes)
        DataArray containing the fatigue equivalent loads of the DLB
        for each sensor and Woehler slope.\n
        Dims: *, m\n
        Coords: *, m

    """
    if isinstance(weight_list, dict):
        weight_list = xr.DataArray(data=list(weight_list.values()),
                                   dims='filename',
                                   coords={'filename': list(weight_list.keys())})
    m_list = eq_loads.m
    DLB_eq_loads = (weight_list*eq_loads**m_list).sum('filename')**(1/m_list)
    return DLB_eq_loads

def get_DLB_eq_loads_from_Markov(markov_matrices, weight_list, m_list, neq=1e7):
    """
    Calculate the fatigue equivalent loads for the whole DLB
    from Markov matrices of individual simulations.

    Parameters
    ----------
    markov_matrices : xarray.DataArray (Nsimulations x * x Nbins x 2)
        DataArray containing number of cycles and load amplitude 
        for each sensor for each simulation.\n
        Dims: filename, *, bin, (cycles, amplitude)\n
        Coords: filename, *
    weight_list : dict or xarray.DataArray
        Dictionary or DataArray containing the weight (real time / simulation time) for each simulation
    m_list : list (Nwoehlerslopes)
        List containing the different woehler slopes
    neq : int or float, optional
        Number of equivalent load cycles. The default is 1e7.

    Returns
    -------
    xarray.DataArray (Nsensors x Ndirections x Nwoehlerslopes)
        DataArray containing the fatigue equivalent loads of the DLB
        for each sensor and Woehler slope.\n
        Dims: *, m\n
        Coords: *, m

    """
    eq_loads = eq_loads_from_Markov(markov_matrix=markov_matrices,
                                    m_list=m_list,
                                    neq=neq)
    eq_loads = xr.DataArray(eq_loads)
    eq_loads = eq_loads.rename({f'dim_{i}': markov_matrices.dims[i]
                                for i in range(len(markov_matrices.dims) - 2)})
    eq_loads = eq_loads.rename({eq_loads.dims[-1]: 'm'})
    eq_loads = eq_loads.assign_coords(markov_matrices.coords)
    eq_loads.coords['m'] = m_list
    DLB_eq_loads = get_DLB_eq_loads(eq_loads, weight_list, neq)
    return DLB_eq_loads

def mean_upperhalf(group):
    """
    Calculate the mean of the values in the upper half of a set. To be
    used as the metric for some DLCs.

    Parameters
    ----------
    group : array
        Array containing the maximum values of a given sensor for each 
        simulation in a group

    Returns
    -------
    float
        The mean value of the upper half of simulations

    """
    upperhalf = sorted(group, reverse=True)[0:int(len(group)/2)]
    return np.mean(upperhalf)