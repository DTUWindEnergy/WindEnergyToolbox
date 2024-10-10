import os
import re
import xarray as xr
from copy import copy
from wetb.fatigue_tools.fatigue import eq_loads_from_Markov

def get_DLB_extreme_loads(extreme_loads, regex_list, metric_list, safety_factor_list, sensor_list):
    """
    Calculate the extreme loads for the whole DLB.

    Parameters
    ----------
    extreme_loads : DataArray (Nsimulations x Nsensors x Ndirections)
        DataArray containing extreme values for each simulation
        for each sensor in sensors_info in each direction in angles.
        It matches the output from collect_postproc\n
        Dims: filename, sensor_name, angle\n
        Coords: filename, sensor_name, angle, sensor_unit
    regex_list : dict
        Dictionary containing the regular expression for grouping simulations
        for each DLC.
    metric_list : dict
        Dictionary containing the function to be applied to each group of simulations
        for each DLC.
    safety_factor_list : dict
        Dictionary containing the safety factor to be applied to each group of simulations
        for each DLC.
    sensor_list : list of tuples (Nsensors)
        List of 3-element tuples containing the name of a load sensor and the indices of 2 of their components.\n
        Example: [('Tower base shear force', 0, 1), ('Blade 1 root bending moment', 9, 10)]

    Returns
    -------
    DataArray (Nsensors x Ndirections x 3)
        DataArray containing the extreme loads of the DLB, as well as the DLC
        and group of simulations driving each load.\n
        Dims: sensor_name, angle, (value, dlc, group)\n
        Coords: sensor_name, angle, sensor_unit

    """
    max_loads_per_simulation = {}
    for i in range(extreme_loads.shape[0]):
        file = os.path.basename(extreme_loads[i].coords['filename'].values[()])
        for DLC, regex in regex_list.items():
            match = re.match(regex, os.path.basename(file))
            if match is None:
                continue
            try:
                max_loads_per_simulation[DLC][match.group(0)][file] = []
            except:
                try:
                    max_loads_per_simulation[DLC][match.group(0)] = {}
                    max_loads_per_simulation[DLC][match.group(0)][file] = []
                except:
                    max_loads_per_simulation[DLC] = {}
                    max_loads_per_simulation[DLC][match.group(0)] = {}
                    max_loads_per_simulation[DLC][match.group(0)][file] = []
            break
        for sensor, i_x, i_y in sensor_list:
            max_loads_per_simulation[DLC][match.group(0)][file].append(extreme_loads[i].sel(sensor_name=sensor))
    extreme_loads_per_group = {}
    for DLC, groups in max_loads_per_simulation.items():
        extreme_loads_per_group[DLC] = {}
        for group, sims in groups.items():
            extreme_loads_per_group[DLC][group] = {}
            for s in range(len(sensor_list)):
                extreme_loads_per_group[DLC][group][sensor_list[s][0]] = {}
                for a in range(len(extreme_loads.coords['angle'].values)):
                    extreme_loads_per_group[DLC][group][sensor_list[s][0]][extreme_loads.coords['angle'].values[a]] = metric_list[DLC]([sim[s][a] for sim in sims.values()])*safety_factor_list[DLC]           
    DLB_extreme_loads = []
    for s in range(len(sensor_list)):
        DLB_extreme_loads.append([])
        for a in range(len(extreme_loads.coords['angle'].values)):
            max_load = 0
            for DLC, groups in extreme_loads_per_group.items():
                for group in groups.keys():
                    if groups[group][sensor_list[s][0]][extreme_loads.coords['angle'].values[a]] > max_load:
                        max_load = copy(groups[group][sensor_list[s][0]][extreme_loads.coords['angle'].values[a]])
                        max_load_dlc = copy(DLC)
                        max_load_group = copy(group)
            # DLB_extreme_loads[-1].append([max_load, max_load_dlc, max_load_group])
            DLB_extreme_loads[-1].append([max_load, max_load_dlc, max_load_group])
    data = DLB_extreme_loads
    dims = ['sensor_name', 'angle', '(value, dlc, group)']
    coords = {'sensor_name': extreme_loads.coords['sensor_name'],
              'angle': extreme_loads.coords['angle'],
              'sensor_unit': extreme_loads.coords['sensor_unit'],
               }
    return xr.DataArray(data=data, dims=dims, coords=coords)

def get_DLB_fatigue_loads(markov_matrices, weight_list, m_list, neq=1e7):
    """
    Calculate the fatigue loads for the whole DLB.

    Parameters
    ----------
    markov_matrices : DataArray (Nsimulations x Nsensors x Ndirections x Nbins x 2)
        DataArray containing directional number of cycles and load amplitude 
        for each simulation for each sensor in sensors_info,
        for each direction in angles and for each bin in no_bins.\n
        Dims: filename, sensor_name, angle, bin, (cycles, amplitude)\n
        Coords: filename, sensor_name, angle, sensor_unit
    weight_list : dict
        Dictionary containing the weight (real time / simulation time) for each simulation
    m_list : list (Nwoehlerslopes)
        List containing the different woehler slopes
    neq : int or float, optional
        Number of equivalent load cycles. The default is 1e7.

    Returns
    -------
    DataArray (Nsensors x Ndirections x Nwoehlerslopes)
        DataArray containing the fatigue loads of the DLB for each sensor,
        direction and Woehler slope.\n
        Dims: sensor_name, angle, m\n
        Coords: sensor_name, angle, m, sensor_unit

    """
    individual_eq_loads = eq_loads_from_Markov(markov_matrix=markov_matrices,
                                               m_list=m_list,
                                               neq=neq)
    DLB_fatigue_loads = []
    for i in range(markov_matrices.shape[1]):
        DLB_fatigue_loads.append([])
        for j in range(markov_matrices.shape[2]):
            DLB_fatigue_loads[-1].append([])
            for k in range(len(m_list)):
                m = m_list[k]
                eq_load = 0
                for l in range(markov_matrices.shape[0]):
                    file = os.path.basename(markov_matrices[l, 0, 0, 0, 0].coords['filename'].values[()])[:-5]
                    eq_load += weight_list[file]*individual_eq_loads[l, i, j, k]**m
                DLB_fatigue_loads[-1][-1].append(eq_load**(1/m))
    data = DLB_fatigue_loads
    dims = ['sensor_name', 'angle', 'm']
    coords = {'sensor_name': markov_matrices.coords['sensor_name'],
              'angle': markov_matrices.coords['angle'],
              'm': m_list,
              'sensor_unit': markov_matrices.coords['sensor_unit'],
               }
    return xr.DataArray(data=data, dims=dims, coords=coords) 