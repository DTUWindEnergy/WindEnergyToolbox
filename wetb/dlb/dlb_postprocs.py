import numpy as np
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
    extreme_loads : xarray.DataArray (Nsimulations x Nsensors x Ndirections)
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
    markov_matrices : xarray.DataArray (Nsimulations x Nsensors x Ndirections x Nbins x 2)
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
    xarray.DataArray (Nsensors x Ndirections x Nwoehlerslopes)
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
                    file = os.path.basename(markov_matrices[l, 0, 0, 0, 0].coords['filename'].values[()]).replace('.hdf5', '')
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

def get_weight_dict(file_list,
                    lifetime,
                    Vin,
                    Vr,
                    Vout,
                    Vref=None,
                    Vstep=2,
                    waves=False,
                    prob_dataarray=None,
                    wsp_list=None,
                    wdir_weight_dict={0: 1},
                    wvdir_weight_dict={0: 1},
                    DLC_scaling_factors_per_wsp=None,
                    DLC_scaling_factors_per_yaw={'DLC12': {0: 0.5, 10: 0.25, 350: 0.25},
                                                 'DLC24': {20: 0.5, 340: 0.5},
                                                 'DLC64': {8: 0.5, 352: 0.5}},
                    n_seeds={'DLC12': 6,
                             'DLC24': 3,
                             'DLC64': 6},
                    n_events=None,
                    sim_time={'DLC12': 600,
                              'DLC24': 600,
                              'DLC31': 100,
                              'DLC41': 100,
                              'DLC64': 600},
                    scaling_factor_DLC64_Vin_Vout=0.025):
    """
    Calculate the dictionary containing the weight for each simulation for the
    calculation of damage equivalent loads of the whole DLB.

    Parameters
    ----------
    file_list : list (Nsimulations)
        List containing the paths to all simulations 
    lifetime : int or float
        Turbine's lifetime in years
    Vin: int or float
        Cut-in wind speed
    Vr: int or float
        Rated wind speed
    Vout: int or float
        Cut-out wind speed
    Vref: int or float, optional
        Reference wind speed. Only used if prob_dataarray is not passed. 
        The default is None.
    Vstep: int or float, optional
        Step between the different wind speeds. Only used if 
        prob_dataarray or wsp_list are not passed. The default is 2.
    waves: bool, optional
        Whether there are waves or not. The default is False 
    prob_dataarray: xarray.DataArray, optional
        DataArray containing the probability of each combination of wind speed 
        and wind direction (and wave direction if waves=True). The default is None.
    wsp_list: list, optional
        List containing all simulated wind speeds. Only used if prob_dataarray is not passed.
        The default is None.
    wdir_weight_dict: dict, optional
        Dictionary containing the weights for each wind direction. Only used if
        prob_dataarray is not passed. The default is {0: 1}.
    wvdir_weight_dict: dict, optional
        Dictionary containing the weights for each wave direction. Only used if
        prob_dataarray is not passed and waves is True. The default is {0: 1}.
    DLC_scaling_factors_per_wsp: dict, optional
        Dictionary containing the scaling factors per DLC and windspeed, i.e. 
        {'DLC1': {v1: sf11, v2: sf12, ...}, 'DLC2': {v1: sf21, v2: sf22, ...}, ...}.
        The default is None, in which case it is assumed that DLC64 takes 2.5% of time
        between Vin and Vout and DLC24 takes 50 hours per year.
    DLC_scaling_factors_per_yaw: dict, optional
        Dictionary containing the scaling factors per DLC and yaw misalignment.
        The default is {'DLC12': {0: 0.5, 10: 0.25, 350: 0.25},
                        'DLC24': {20: 0.5, 340: 0.5},
                        'DLC64': {8: 0.5, 352: 0.5}}.
    n_seeds: dict, optional
        Dictionary containing the number of seeds per DLC per wind speed and wind direction.
        The default is {'DLC12': 6,
                        'DLC24': 3,
                        'DLC64': 6}.
    n_events: dict, optional
        Dictionary containing the number of events per DLC per windspeed.
        The default is {'DLC31': {Vin: 1000, Vr: 50, Vout: 50},
                        'DLC41': {Vin: 1000, Vr: 50, Vout: 50}}.
    sim_time: dict, optional
        Dictionary containing the simulation time per DLC.
        The default is {'DLC12': 600,
                        'DLC24': 600,
                        'DLC31': 100,
                        'DLC41': 100,
                        'DLC64': 600} 
    scaling_factor_DLC64_Vin_Vout: int or float, optional
        Percentage of time that DLC64 takes between Vin and Vout. Only used if
        DLC_scaling_factors_per_wsp is not passed.
        The default is 0.025.
            
    Returns
    -------
    dict (Nsimulations)
        Dictionary containing the weight for each simulation.

    """
    if prob_dataarray is None:
        if wsp_list is None:
            wsp_list = list(range(Vin, int(0.7*Vref) + 1, Vstep))
        coords = {'wsp': wsp_list}
        from wetb.dlc.high_level import Weibull_IEC
        prob_data = Weibull_IEC(Vref=Vref, Vhub_lst=wsp_list)
        prob_data = prob_data[:, np.newaxis]*np.array(list(wdir_weight_dict.values()))[np.newaxis, :]
        coords['wdir'] = list(wdir_weight_dict.keys())
        if waves:
            prob_data = prob_data[:, :, np.newaxis]*np.array(list(wvdir_weight_dict.values()))[np.newaxis, :]
            coords['wvdir'] =  list(wvdir_weight_dict.keys())
            prob_dataarray = xr.DataArray(data=prob_data,
                                          dims=('wsp', 'wdir', 'wvdir'),
                                          coords=coords)
        else:
            prob_dataarray = xr.DataArray(data=prob_data,
                                          dims=('wsp', 'wdir'),
                                          coords=coords)
    else:
        wsp_list = list(prob_dataarray.wsp.values)
    
    if DLC_scaling_factors_per_wsp is None:
        prob_Vin_Vout = prob_dataarray.where((prob_dataarray.wsp >= Vin) & (prob_dataarray.wsp <= Vout), drop=True).sum().values[()]
        scaling_factor_DLC24 = (50/prob_Vin_Vout)/(365.25*24)
        scaling_factor_DLC12 = 1 - scaling_factor_DLC24 - scaling_factor_DLC64_Vin_Vout                
        DLC_scaling_factors_per_wsp = {'DLC12': {wsp: scaling_factor_DLC12 if Vin <= wsp <= Vout else 0 for wsp in wsp_list},
                                       'DLC24': {wsp: scaling_factor_DLC24 if Vin <= wsp <= Vout else 0 for wsp in wsp_list},
                                       'DLC64': {wsp: scaling_factor_DLC64_Vin_Vout if Vin <= wsp <= Vout else 1 for wsp in wsp_list}}
       
    if n_events is None:
        n_events = {'DLC31': {Vin: 1000, Vr: 50, Vout: 50},
                    'DLC41': {Vin: 1000, Vr: 50, Vout: 50}} 
    
    file_hour_dict = {}
    for file in file_list:
        file = os.path.basename(file)
        i1 = file.find('DLC')
        i2 = file.find('_', i1)
        DLC = file[i1:i2]
        i3 = file.find('wsp')
        i4 = file.find('_', i3)
        wsp = int(file[i3 + 3:i4])
        if DLC in ['DLC12', 'DLC24', 'DLC64']:
            i5 = file.find('wdir')
            i6 = file.find('_', i5)
            wdir = int(file[i5 + 4:i6])
            i7 = file.find('yaw')
            i8 = file.find('_', i7)
            yaw = int(file[i7 + 3:i8])
            if waves:
                i9 = file.find('wvdir')
                i10 = file.find('_', i9)
                wvdir = int(file[i9 + 5:i10])
                file_hour_dict[file] = (365.25*24*prob_dataarray.sel(wsp=wsp).sel(wdir=wdir).sel(wvdir=wvdir).values[()]*
                DLC_scaling_factors_per_wsp[DLC][wsp]*DLC_scaling_factors_per_yaw[DLC][yaw]/n_seeds[DLC])
            else:
                file_hour_dict[file] = (365.25*24*prob_dataarray.sel(wsp=wsp).sel(wdir=wdir).values[()]*
                DLC_scaling_factors_per_wsp[DLC][wsp]*DLC_scaling_factors_per_yaw[DLC][yaw]/n_seeds[DLC])
        elif DLC in ['DLC31', 'DLC41']:
            file_hour_dict[file] = n_events[DLC][wsp]*sim_time[DLC]/3600
    
    weight_dict = {file: lifetime*hours*3600/sim_time[DLC] for file, hours in file_hour_dict.items()}
    return weight_dict