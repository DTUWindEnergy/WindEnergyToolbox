import numpy as np
import os
import re
import xarray as xr
from copy import copy
from wetb.fatigue_tools.fatigue import eq_loads_from_Markov

def nc_to_dataarray(nc_file):
    dataarray = xr.open_dataset(nc_file).to_dataarray().squeeze('variable')
    if 'sensor_unit' in dataarray.coords:
        dataarray = dataarray.assign_coords(sensor_unit=dataarray["sensor_unit"].astype(str))
    if 'sensor_description' in dataarray.coords:
        dataarray = dataarray.assign_coords(sensor_description=dataarray["sensor_description"].astype(str))
        dataarray = add_coords_from_sensor_description(dataarray)
    return dataarray

def get_params(filename, regex):
    match = re.match(regex, os.path.basename(filename))
    if match is not None:
        return match.groups()
    else:
        return np.nan

def add_coords_from_filename(dataarray, params, regex, formats=None):
    coords = xr.apply_ufunc(get_params,
                            dataarray.coords['filename'],
                            kwargs={'regex': regex},
                            input_core_dims=[[]],
                            output_core_dims=len(params)*[[]],
                            vectorize=True)
    if formats is None:
        for i in range(len(params)):
            dataarray.coords[params[i]] = ('filename', coords[i].values)
    else:
        for i in range(len(params)):
            dataarray.coords[params[i]] = ('filename', [formats[i](v) for v in coords[i].values])
    return dataarray

def get_load_info(sensor_description):
    if sensor_description.startswith('Force_intp'):
        load = sensor_description[12:14]
        mbdy_end = sensor_description.find(' ', 20)
        mbdy = sensor_description[20:mbdy_end]
        s_start = sensor_description.find('s=', mbdy_end) + 3
        s_end = s_start + 6
        s = float(sensor_description[s_start: s_end])
        ss_start = sensor_description.find('s/S=', mbdy_end) + 5
        ss_end = ss_start + 6
        ss = float(sensor_description[ss_start: ss_end])
        coo_start = sensor_description.find('coo:', ss_end) + 5
        coo_end = sensor_description.find(' ', coo_start)
        coo = sensor_description[coo_start:coo_end]
        center_start = sensor_description.find('center:', coo_end) + 7
        center_end = sensor_description.find(' ', center_start)
        center = sensor_description[center_start:center_end]
        return coo, load, mbdy, None, s, ss, center
    elif sensor_description.startswith('Moment_intp'):
        load = sensor_description[13:15]
        mbdy_end = sensor_description.find(' ', 21)
        mbdy = sensor_description[21:mbdy_end]
        s_start = sensor_description.find('s=', mbdy_end) + 3
        s_end = s_start + 6
        s = float(sensor_description[s_start: s_end])
        ss_start = sensor_description.find('s/S=', mbdy_end) + 5
        ss_end = ss_start + 6
        ss = float(sensor_description[ss_start: ss_end])
        coo_start = sensor_description.find('coo:', ss_end) + 5
        coo_end = sensor_description.find(' ', coo_start)
        coo = sensor_description[coo_start:coo_end]
        center_start = sensor_description.find('center:', coo_end) + 7
        center_end = sensor_description.find(' ', center_start)
        center = sensor_description[center_start:center_end]
        return coo, load, mbdy, None, s, ss, center
    elif sensor_description.startswith('Force'):
        load = sensor_description[7:9]
        mbdy_end = sensor_description.find(' ', 15)
        mbdy = sensor_description[15:mbdy_end]
        node_start = sensor_description.find('nodenr:', mbdy_end) + 8
        node_end = node_start + 3
        node = int(sensor_description[node_start:node_end])
        coo_start = sensor_description.find('coo:', node_end) + 5
        coo_end = sensor_description.find(' ', coo_start)
        coo = sensor_description[coo_start:coo_end]
        return coo, load, mbdy, node, None, None, None
    elif sensor_description.startswith('Moment'):
        load = sensor_description[6:8]
        mbdy_end = sensor_description.find(' ', 14)
        mbdy = sensor_description[14:mbdy_end]
        node_start = sensor_description.find('nodenr:', mbdy_end) + 8
        node_end = node_start + 3
        node = int(sensor_description[node_start:node_end])
        coo_start = sensor_description.find('coo:', node_end) + 5
        coo_end = sensor_description.find(' ', coo_start)
        coo = sensor_description[coo_start:coo_end]
        return coo, load, mbdy, node, None, None, None
    else:
        return None, None, None, None, None, None, None

def add_coords_from_sensor_description(dataarray):
    coords = xr.apply_ufunc(get_load_info,
                            dataarray.coords['sensor_description'],
                            input_core_dims=[[]],
                            output_core_dims=[[], [], [], [], [], [], []],
                            vectorize=True)
    coords_labels = ['coo', 'load', 'mbdy', 'node', 's', 's/S', 'center']
    for i in range(len(coords_labels)):
        dataarray.coords[coords_labels[i]] = ('sensor_name', coords[i].values)
    return dataarray

def get_DLC(filename):
    filename = os.path.basename(filename)
    i1 = filename.lower().find('dlc')
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
    return DLB_extreme_loads.transpose(..., 'sensor_name', 'driver', 'load')

def ext_or_cont(ext_and_cont, driver, sensor_description, metric, safety_factor):
    if sensor_description in driver:  
        return metric(ext_and_cont)*safety_factor
    else:
        return np.mean(np.abs(ext_and_cont))*np.sign(ext_and_cont[np.argmax(np.abs(ext_and_cont))])

def get_DLB_ext_and_cont(ext_and_cont, regex_list, metric_list, safety_factor_list):
    """
    Analogue function to get_DLB_extreme_loads, but generalized for any given
    set of sensors and not only the 6 load components of a node.

    Parameters
    ----------
    ext_and_cont : xarray.DataArray (Nsimulations x Nsensors*2 x Nsensors)
        DataArray containing collected data for each simulation of the extreme values
        (max and min) and their contemporaneous values for a given set of sensors.
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
    DataArray (Nsensors*2 x Nsensors)
        DataArray containing the extreme values (max and min) and their contemporaneous
        values for a given set of sensors.

    """
    grouped_simulations = group_simulations(ext_and_cont, regex_list)
    values_by_group_dict = {}
    for group, simulations in grouped_simulations.items():
        DLC = get_DLC(group)
        group_values = xr.apply_ufunc(ext_or_cont,
                                     simulations,
                                     simulations.coords['driver'], 
                                     simulations.coords['sensor_description'], 
                                     kwargs={'metric': metric_list[DLC], 'safety_factor': safety_factor_list[DLC]},
                                     input_core_dims=[['filename'], [], []],
                                     vectorize=True)
        values_by_group_dict[group] = group_values
    values_by_group = xr.concat(list(values_by_group_dict.values()), dim='group')
    values_by_group = values_by_group.drop_vars('variable')
    values_by_group['group'] = list(values_by_group_dict.keys())

    driving_group_indices = []
    for driver in values_by_group.coords['driver'].values:
        sensor_description = driver[:-4]
        metric = driver[-3:]
        if metric == 'max':
            driving_group_indices.append(values_by_group.sel(driver=driver, sensor_description=sensor_description).argmax('group'))
        elif metric == 'min':
            driving_group_indices.append(values_by_group.sel(driver=driver, sensor_description=sensor_description).argmin('group'))
    driving_group_indices = xr.concat(driving_group_indices, dim='driver')
    driving_group_indices = driving_group_indices.drop_vars('sensor_description')
    DLB_ext_and_cont = values_by_group.isel(group=driving_group_indices)
    return DLB_ext_and_cont

def get_values_by_group(dataarray, regex_list, metric_list, safety_factor_list):
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

def get_DLB_directional_extreme_loads(directional_extreme_loads, regex_list, metric_list, safety_factor_list):
    """
    Identic procedure as in get_DLB_extreme_loads, but for maximum loads 
    along each direction.

    Parameters
    ----------
    directional_extreme_loads : xarray.DataArray 
        DataArray containing the collected directional extreme loads of all simulations.
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
    xarray.DataArray
        DataArray containing the maximum values of the DLB, as well as 
        the group of simulations driving each load direction

    """
    group_values = get_values_by_group(directional_extreme_loads, regex_list, metric_list, safety_factor_list)
    
    DLB_directional_extreme_loads = group_values.max('group')
    DLB_directional_extreme_loads.coords['group'] = group_values['group'].isel(group=group_values.argmax('group'))
    return DLB_directional_extreme_loads

def get_DLB_extreme_values(statistics, regex_list, metric_list, safety_factor_list):
    """
    Group sensors maxima and minima of each timeseries by regular expression,
    apply a metric and scale by a safety factor. Identic procedure as in
    get_DLB_extreme_loads, but applycable for any sensor and only extreme values
    are computed, not contemporaneous values of other sensors.

    Parameters
    ----------
    statistics : xarray.DataArray (Nsimulations x *)
        DataArray containing the collected statistics of all simulations.
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
    DataArray (Nsensors x 2)
        DataArray containing the maximum and minimum values of the DLB, as well as 
        the group of simulations driving each sensor

    """
    group_values = get_values_by_group(statistics, regex_list, metric_list, safety_factor_list)
    
    max_values = group_values.sel(statistic='max').max('group')
    max_indices = group_values.sel(statistic='max').argmax('group')
    max_values.coords['group'] = group_values.sel(statistic='max')['group'].isel(group=max_indices)
    
    min_values = group_values.sel(statistic='min').min('group')
    min_indices = group_values.sel(statistic='min').argmin('group')
    min_values.coords['group'] = group_values.sel(statistic='min')['group'].isel(group=min_indices)
    DLB_extreme_values = xr.concat([max_values, min_values], dim='statistic')
    if 'sensor_name' in DLB_extreme_values.dims:
        DLB_extreme_values = DLB_extreme_values.transpose(..., 'sensor_name', 'statistic')
    return DLB_extreme_values

def get_DLB_eq_loads(eq_loads, weight_list):
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
    if 'variable' in DLB_eq_loads.coords:
        DLB_eq_loads = DLB_eq_loads.drop_vars('variable')
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
    DLB_eq_loads = get_DLB_eq_loads(eq_loads, weight_list)
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

def get_weight_list(file_list,
                    n_years,
                    Vin,
                    Vout,
                    Vr,
                    wsp_list,
                    probability,
                    wsp_weights=None,
                    yaw_weights=xr.concat([xr.DataArray(data=[[0.5, 0.25, 0.25]],
                                                        dims=('dlc', 'wdir'),
                                                        coords={'dlc': ['12'], 'wdir': [0, 10, 350]}),
                                           xr.DataArray(data=[[0.5, 0.5]],
                                                        dims=('dlc', 'wdir'),
                                                        coords={'dlc': ['24'], 'wdir': [20, 340]}),
                                           xr.DataArray(data=[[0.5, 0.5]],
                                                        dims=('dlc', 'wdir'),
                                                        coords={'dlc': ['64'], 'wdir': [8, 352]})],
                                           dim='dlc'),
                    n_seeds=xr.DataArray(data=[6, 3, 6],
                                         dims=('dlc'),
                                         coords={'dlc': ['12', '24', '64']}),
                    n_events=None,
                    sim_time=xr.DataArray(data=[600, 600, 100, 100, 600],
                                          dims=('dlc'),
                                          coords={'dlc': ['12', '24', '31', '41', '64']}),
                    weight_DLC64_Vin_Vout=0.025,
                    neq=None):
    """
    Calculate the weights for each simulation for the
    calculation of damage equivalent loads of the whole DLB.

    Parameters
    ----------
    file_list : xr.DataArray
        DataArray containing the paths to all simulations. It must also have
        coordinates for DLC, wind speed, wind direction and/or wave direction
    n_years : int or float
        Turbine's lifetime in years
    Vin: int or float
        Cut-in wind speed
    Vout: int or float
        Cut-out wind speed
    Vr: int or float
        Rated wind speed
    wsp_list: array
        Array containing all simulated wind speeds
    probability: xarray.DataArray
        DataArray containing the probability of each combination of wind speed, 
        wind direction and/or wave direction
    wsp_weights: xarray.DataArray, optional
        DataArray containing the percentage of time each DLC takes for each wind speed.
        The default assumes that DLC64 takes 2.5% of time between Vin and Vout
    yaw_weights: xarray.DataArray, optional
        DataArray containing the percentage of time each yaw misalignment takes for each DLC.
        The default assumes {-10: 0.25, 0: 0.5, 10: 0.25} for DLC12,
        {-20: 0.5, 20: 0.5} for DLC24 and {-8: 0.5, 8: 0.5} for DLC64
    n_seeds: xarray.DataArray, optional
        DataArray containing the number of seeds per DLC per wind speed and wind direction and/or wave direction.
        The default assumes 6 seeds for DLC12, 3 seeds for DLC24 and 6 seeds for DLC64
    n_events: xarray.DataArray, optional
        DataArray containing the number of events per DLC per windspeed.
        The default assumes {Vin: 1000, Vr: 50, Vout: 50} for both DLC31 and DLC41.
    sim_time: xarray.DataArray, optional
        DataArray containing the simulation time per DLC.
        The default assumes 600s for DLC12, DLC24, DLC64 and 100s for DLC31 and DLC41
    weight_DLC64_Vin_Vout: int or float, optional
        Percentage of time that DLC64 takes between Vin and Vout. Only used if
        wsp_weights is not passed.
        The default is 0.025.
    neq : int or float, optional
        Number of equivalent load cycles. If not passed, the output weights
        can ONLY be used when calculating DLB equivalent loads from individual file
        Markov matrices. If passed, the output weights can ONLY be used when 
        calculating DLB equivalent loads from individual equivalent loads (recommended)
            
    Returns
    -------
    xarray.DataArray 
        DataArray containing the weight for each simulation.

    """
    
    if wsp_weights is None:
        weight_DLC24 = 50/(365.25*24)
        weight_DLC12 = 1 - weight_DLC64_Vin_Vout                
        wsp_weights = xr.DataArray(data=[[weight_DLC12 if Vin <= v <= Vout else 0 for v in wsp_list],
                                         [weight_DLC24 if Vin <= v <= Vout else 0 for v in wsp_list],
                                         [weight_DLC64_Vin_Vout if Vin <= v <= Vout else 1 for v in wsp_list]],
                                   dims=('dlc', 'wsp'),
                                   coords={'dlc': ['12', '24', '64'],
                                           'wsp': wsp_list})       
    if n_events is None:
        n_events = xr.DataArray(data=[[1000, 50, 50], 
                                      [1000, 50, 50]],
                                dims=('dlc', 'wsp'),
                                coords={'dlc': ['31', '41'],
                                        'wsp': [Vin, Vr, Vout]})
    
    lifetime = n_years*365.25*24*3600 # convert to seconds
    weight_list = (lifetime*probability*wsp_weights*yaw_weights/n_seeds/sim_time).combine_first(n_years*n_events)
    if neq is not None: # correct by sim_time/neq so it can be used for individual equivalent loads
        weight_list = weight_list*sim_time/neq
    
    # rearrange weights from parameters (DLC, wsp, wdir) to actual filenames
    # TO-DO: allow wvdir as well
    def get_weight_list_per_filename(file_list, dlc, wsp, wdir, weight_list):
        return weight_list.sel(dlc=dlc, wsp=wsp, wdir=wdir)    
    
    weight_list = xr.apply_ufunc(get_weight_list_per_filename,
                       file_list,
                       file_list['dlc'],
                       file_list['wsp'],
                       file_list['wdir'],
                       kwargs={'weight_list': weight_list},
                       vectorize=True)
    
    return weight_list
