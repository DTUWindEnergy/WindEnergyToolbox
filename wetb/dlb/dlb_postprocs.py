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
    Calculate the extreme values for the whole DLB.

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
    weight_list : dict
        Dictionary containing the weight (real time / simulation time) for each simulation
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
    weight_list : dict
        Dictionary containing the weight (real time / simulation time) for each simulation
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
    eq_loads = eq_loads.drop_vars('variable')
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