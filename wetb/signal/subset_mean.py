'''
Created on 24/06/2016

@author: MMPE
'''

import numpy as np
import unittest
from wetb.utils.geometry import rpm2rads
from _collections import deque
from tables.tests.test_index_backcompat import Indexes2_0TestCase


def power_mean(power, trigger_indexes, I, rotor_speed, time, air_density=1.225, rotor_speed_mean_samples=1) :
    """Calculate the density normalized mean power, taking acceleration of the rotor into account

    Parameters
    ---------
    Power : array_like
        Power [W]
    trigger_indexes : array_like
        Trigger indexes
    I : float
        Rotor inerti [kg m^2]
    rotor_speed : array_like
        Rotor speed [rad/s]
    time : array_like
        time [s]
    air_density : int, float or array_like, optional
        Air density.
    rotor_speed_mean_samples : int
        To reduce the effect of noise, the mean of a number of rotor speed samples can be used

    Returns
    -------
    mean power including power used to (de)accelerate rotor

    Examples:
    ---------
    turbine_power_mean = lambda power, triggers : power_mean(power, triggers, I=2.5E7, rot_speed, time, rho)
    trigger_indexes = time_trigger(time,30)
    wsp_mean, power_mean = subset_mean([wsp, power],trigger_indexes,mean_func={1:turbine_power_mean})
    """
    if rotor_speed_mean_samples == 1:
        rs1 = rotor_speed[trigger_indexes[:-1]]
        rs2 = rotor_speed[trigger_indexes[1:] - 1]
    else:
        rs = np.array([rotor_speed[max(i - rotor_speed_mean_samples, 0):i - 1 + rotor_speed_mean_samples].mean() for i in trigger_indexes])
        rs1 = rs[:-1]
        rs2 = rs[1:]


    power = np.array([np.nanmean(power[i1:i2], 0) for i1, i2 in zip(trigger_indexes[:-1].tolist(), trigger_indexes[1:].tolist())])
    if isinstance(air_density, (int, float)):
        if air_density != 1.225:
            power = power / air_density * 1.225
    else:
        air_density = np.array([np.nanmean(air_density[i1:i2], 0) for i1, i2 in zip(trigger_indexes[:-1].tolist(), trigger_indexes[1:].tolist())])
        power = power / air_density * 1.225
    return power + 1 / 2 * I * (rs2 ** 2 - rs1 ** 2) / (time[trigger_indexes[1:] - 1] - time[trigger_indexes[:-1]])

def power_mean_func_kW(I, rotor_speed, time, air_density=1.225, rotor_speed_mean_samples=1) :
    """Return a power mean function [kW] used to Calculate the density normalized mean power, taking acceleration of the rotor into account

    Parameters
    ---------
    I : float
        Rotor inerti [kg m^2]
    rotor_speed : array_like
        Rotor speed [rad/s]
    time : array_like
        time [s]
    air_density : int, float or array_like, optional
        Air density.
    rotor_speed_mean_samples : int
        To reduce the effect of noise, the mean of a number of rotor speed samples can be used

    Returns
    -------
    mean power function

    Examples:
    ---------
    turbine_power_mean = power_mean_func_kW(power, triggers, I=2.5E7, rot_speed, time, rho)
    trigger_indexes = time_trigger(time,30)
    wsp_mean, power_mean = subset_mean([wsp, power],trigger_indexes,mean_func={1:turbine_power_mean})
    """
    def mean_power(power, trigger_indexes):
        return power_mean(power * 1000, trigger_indexes, I, rotor_speed, time , air_density, rotor_speed_mean_samples) / 1000
    return mean_power


def subset_mean(data, trigger_indexes, mean_func={}):
    if isinstance(data, list):
        data = np.array(data).T
    if len(data.shape)==1:
        no_sensors = 1
    else:
        no_sensors = data.shape[1]
    if isinstance(trigger_indexes[0], tuple):
        triggers = np.array(trigger_indexes)
        steps = np.diff(triggers[:, 0])
        lengths = np.diff(triggers)[:, 0]
        if np.all(steps == steps[0]) and np.all(lengths == lengths[0]):
            subset_mean = np.mean(np.r_[data.reshape(data.shape[0],no_sensors),np.empty((steps[0],no_sensors))+np.nan][triggers[0][0]:triggers.shape[0] * steps[0] + triggers[0][0]].reshape(triggers.shape[0], steps[0], no_sensors)[:, :lengths[0]], 1)
        else:
            subset_mean = np.array([np.mean(data[i1:i2], 0) for i1, i2 in trigger_indexes])
        for index, func in mean_func.items():
            att = data[:, index]
            subset_mean[:, index] = func(att, trigger_indexes)
    else:
        steps = np.diff(trigger_indexes)

        if np.all(steps == steps[0]):
            #equal distance
            subset_mean = np.mean(data[trigger_indexes[0]:trigger_indexes[-1]].reshape([ len(trigger_indexes) - 1, steps[0], data.shape[1]]), 1)
        else:
            subset_mean = np.array([np.mean(data[i1:i2], 0) for i1, i2 in zip(trigger_indexes[:-1].tolist(), trigger_indexes[1:].tolist())])
        for index, func in mean_func.items():
            att = data[:, index]
            subset_mean[:, index] = func(att, trigger_indexes)
    if len(data.shape)==1 and len(subset_mean.shape)==2:
        return subset_mean[:,0]
    else:
        return subset_mean


def cycle_trigger(values, trigger_value=None, step=1, ascending=True, tolerance=0):
    values = np.array(values)
    if trigger_value is None:
        r = values.max() - values.min()
        values = (values[:] - r / 2) % r
        trigger_value = r / 2
    if ascending:
        return np.where((values[1:] > trigger_value + tolerance) & (values[:-1] <= trigger_value - tolerance))[0][::step]
    else:
        return np.where((values[1:] < trigger_value - tolerance) & (values[:-1] >= trigger_value + tolerance))[0][::step]

def revolution_trigger(rotor_position, sample_frq, rotor_speed, max_no_round_diff=1):
    """Returns one index per revolution (minimum rotor position)
    
    Parameters
    ----------
    rotor_position : array_like
        Rotor position [deg] (0-360)
    sample_frq : int or float
        Sample frequency [Hz]
    rotor_speed : array_like
        Rotor speed [RPM]
        
    Returns
    -------
    nd_array : Array of indexes
    """
    if isinstance(rotor_speed, (float, int)):
        rotor_speed = np.ones_like(rotor_position)*rotor_speed
    deg_per_sample = rotor_speed*360/60/sample_frq
    sample_per_round = 1/(rotor_speed/60/sample_frq)
    thresshold = deg_per_sample.max()*2
    
    nround_rotor_speed = np.nansum(rotor_speed/60/sample_frq)
    
    mod = [v for v in [5,10,30,60,90] if v>thresshold][0]
    
    nround_rotor_position = np.nansum(np.diff(rotor_position)%mod)/360
    #assert abs(nround_rotor_position-nround_rotor_speed)<max_no_round_diff, "No of rounds from rotor_position (%.2f) mismatch with no_rounds from rotor_speed (%.2f)"%(nround_rotor_position, nround_rotor_speed)
    #print (nround_rotor_position, nround_rotor_speed)
    
    rp = np.array(rotor_position).copy()
    #filter degree increase > thresshold
    rp[np.r_[False, np.diff(rp)>thresshold]] = 180
    
    upper_indexes = np.where((rp[:-1]>(360-thresshold))&(rp[1:]<(360-thresshold)))[0]
    lower_indexes = np.where((rp[:-1]>thresshold)&(rp[1:]<thresshold))[0] +1 
    
    # Best lower is the first lower after upper
    best_lower = lower_indexes[np.searchsorted(lower_indexes, upper_indexes)]
    upper2lower = best_lower - upper_indexes
    best_lower = best_lower[upper2lower<upper2lower.mean()*2]
    #max_dist_error = max([np.abs((i2-i1)- np.mean(sample_per_round[i1:i2])) for i1,i2 in zip(best_lower[:-1], best_lower[1:])])
    #assert max_dist_error < sample_frq/5, max_dist_error
    return best_lower
    

def revolution_trigger_old(values, rpm_dt=None, dmin=5, dmax=10, ):
    """Return indexes where values are > max(values)-dmin and decreases more than dmax
    If RPM and time step is provided, triggers steps < time of 1rpm is removed   
    
    Parameters
    ---------
    values : array_like
        Position signal (e.g. rotor position)
    rpm_dt : tuple(array_like, float), optional
        - rpm : RPM signal
        - dt : time step between samples
    dmin : int or float, optional
        Minimum normal position increase between samples
    dmax : float or int, optional
        Maximum normal position increase between samples 
            
    
    Returns
    -------
    trigger indexes 
        [i1,i2,...,in] if rpm_dt is not provided
        [(start1,stop1),(start2,stop2),...,(startn, stopn)] if rpm_dt is provided
    """
    
    values = np.array(values)
    indexes = np.where((np.diff(values)<-dmax)&(values[:-1]>values.max()-dmax))[0]
    
    if rpm_dt is None:
        return indexes
    else:
        index_pairs = []
        rpm, dt = rpm_dt
        d_deg = rpm *360/60*dt
        cum_d_deg = np.cumsum(d_deg)
        lbound, ubound = values.max()-dmax, values.max()+dmax
        index_pairs = [(i1,i2) for i1,i2, deg in zip(indexes[:-1], indexes[1:], cum_d_deg[indexes[1:]-1]-cum_d_deg[indexes[:-1]]) 
                       if deg > lbound and deg<ubound]
        return index_pairs
    
    
def time_trigger(time, step, start=None, stop=None):
    if start is None:
        start = time[0]
    decimals = int(np.ceil(np.log10(1 / np.nanmin(np.diff(time)))))
    time = np.round(time - start, decimals)


    steps = np.round(np.diff(time), decimals)
    if np.sum(steps == steps[0])/len(time)>.99: #np.all(steps == steps[0]):
        # equal time step
        time = np.r_[time, time[-1] + max(set(steps), key=list(steps).count)]
    if stop is None:
        stop = time[~np.isnan(time)][-1]
    else:
        stop -= start
    epsilon = 10 ** -(decimals + 2)
    return np.where(((time % step < epsilon) | (time % step > step - epsilon)) & (time >= 0) & (time <= stop))[0]


def non_nan_index_trigger(sensor, step):
    trigger = []
    i = 0
    nan_indexes = deque(np.where(np.isnan(sensor))[0].tolist() + [len(sensor)])
    while i + step <= sensor.shape[0]:
        if i+step<=nan_indexes[0]:
            trigger.append((i,i+step))
            i+=step
        else:
            i = nan_indexes.popleft()+1
    return trigger

