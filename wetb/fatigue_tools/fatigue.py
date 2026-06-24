# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:28:30 2026

@author: nicgo

There are 2 available rainflow counting methods (See documentation in top of methods):
- 'rainflow_windap': (Described in "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads",
                      2. edition 1990, Appendix A)
or
- 'rainflow_astm' (based on the c-implementation by Adam Nieslony found at the MATLAB Central File Exchange
                   http://www.mathworks.com/matlabcentral/fileexchange/3026)

The low level functions of this submodule are rainflow_windap, rainflow_astm, shifted_Goodman_diagram and palmgren_miner.
eq_load, eq_load_from_cycles and cycle_matrix are wrappers built on top that the user should usually use instead according to their needs.
"""

import numpy as np
import warnings
from wetb.fatigue_tools.rainflowcounting.rainflowcount import rainflow_windap, rainflow_astm

def shifted_Goodman_diagram(ampls, means, L_u_t, L_u_c):
    '''
    Perform mean load correction through a shifted Goodman diagram
    according to DNV ST-0376:2024 (2024)

    Parameters
    ----------
    ampls : array-like
        Array containing the amplitudes of each load cycle.
    means : array-like
        Array containing the means of each load cycle.
    L_u_t : float
        Ultimate load for tension.
    L_u_c : float
        Ultimate load for compression

    Returns
    -------
    ampls_mlc : array-like
        Array containing the corrected amplitudes of each load cycle.
    '''
    L_u_avg = abs(L_u_t - L_u_c) / 2
    L_u_mid = (L_u_t + L_u_c) / 2
    ampls_mlc = ampls * (L_u_avg - abs(L_u_mid)) / (L_u_avg - abs(means - L_u_mid))
    return ampls_mlc

def palmgren_miner(ampls, cycles, m, neq):
    '''
    Compute damage equivalent load through Palmgren Miner rule.

    Parameters
    ----------
    ampls : array-like
        Array containing the amplitudes.
    cycles : array-like
        Array containing the number of cycles.
    m : int, float or array-like
        Woehler exponent.
    neq : int, float or array-like
        Equivalent number of cycles.

    Returns
    -------
    dels : array
        Array containing the damage equivalent loads. 
    '''
    dels = np.zeros((len(np.atleast_1d(neq)), len(np.atleast_1d(m))))
    for i, _neq in enumerate(np.atleast_1d(neq)):
        for j, _m in enumerate(np.atleast_1d(m)):
            dels[i, j] = (np.nansum(cycles * ampls ** _m) / _neq) ** (1 / _m)
    return dels

def eq_load(signals,
            rainflow_func=rainflow_windap,
            mlc_func=None,
            m=[3, 4, 6, 8, 10, 12],
            neq=1,
            **kwargs):
    '''
    Wrapper function to compute a damage equivalent load. It performs a 
    rainflow counting, followed by mean load correction (if requested)
    and finally applies damage accumulation through Palmgren Miner rule.

    Parameters
    ----------
    signals : array-like or list of tuples
        - If array-like, the raw signal.
        - If list of tuples, list of (weight, signal), e.g. [(0.1 , sig), (0.8 , sig2), (0.1 , sig3)].
    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap).
    mlc_func : func, optional
        Mean load correction function to use. The default is None (do not perform mean load correction).
    m : int, float or array-like
        Woehler exponent. The default is [3, 4, 6, 8, 10, 12].
    neq : int, float or array-like
        Equivalent number of cycles. The default is 1.
    **kwargs : 
        Additional keyword arguments other than ampls and means needed by mlc_func.

    Returns
    -------
    dels : array
        Array containing the damage equivalent loads
    '''
    # Rainflow counting
    if isinstance(signals[0], tuple):
        weights, ampls, means = np.array([(np.ones_like(ampl) * weight, ampl, mean) for weight, signal in signals
                                          for ampl, mean in rainflow_func(signal[:]).T], dtype=np.float64).T
    else:
        ampls, means = rainflow_func(signals[:])
        weights = np.ones_like(ampls)
        
    # Mean load correction (if requested)
    if mlc_func is not None:
        ampls = mlc_func(ampls, means, **kwargs)

        
    # Damage equivalent load calculation
    cycles = np.histogram(ampls, bins=np.concat((np.unique(ampls), np.array([np.max(ampls) + 1]))), weights=weights)[0]
    cycles = cycles / 2  # to get full cycles
    ampls = np.unique(ampls)
    dels = palmgren_miner(ampls, cycles, m, neq)    
    return dels

def eq_load_from_cycles(ampls,
                        cycles,
                        mlc_func=None,
                        m=[3, 4, 6, 8, 10, 12],
                        neq=1,
                        **kwargs):
    '''
    Wrapper function to compute a damage equivalent load. Equivalent to 
    eq_load except it skips rainflow counting as the load amplitudes and cycles
    are already passed.

    Parameters
    ----------
    ampls : array-like
        Array containing load cycle amplitudes.
    cycles: array-like
        Array containing number of load cycles.
    mlc_func : func, optional
        Mean load correction function to use. The default is None (do not perform mean load correction).
    m : int, float or array-like
        Woehler exponent. The default is [3, 4, 6, 8, 10, 12].
    neq : int, float or array-like
        Equivalent number of cycles. The default is 1.
    **kwargs : 
        Additional keyword arguments other than ampls needed by mlc_func.

    Returns
    -------
    dels : array
        Array containing the damage equivalent loads
    '''        
    # Mean load correction (if requested)
    if mlc_func is not None:
        ampls = mlc_func(ampls, **kwargs)
        
    # Damage equivalent load calculation
    dels = palmgren_miner(ampls, cycles, m, neq)    
    return dels

def cycle_matrix(signals,
                 rainflow_func=rainflow_windap,
                 ampl_bins=10,
                 mean_bins=10,
                 bin_ampl_from_zero=False):
    '''
    Wrapper function to compute a Markov load cycle matrix. It performs
    a rainflow counting, and then builds the matrix by binning the cycles
    by amplitude and mean.

    Parameters
    ----------
    signals : array-like or list of tuples
        - If array-like, the raw signal.
        - If list of tuples, list of (weight, signal), e.g. [(0.1 , sig), (0.8 , sig2), (0.1 , sig3)].
    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap).
    ampl_bins : int or array-like, optional
        - If int, Number of amplitude value bins (default is 10).
        - If array-like, the bin edges for amplitude.
    mean_bins : int or array-like, optional
        - If int, Number of mean value bins (default is 10).
        - If array-like, the bin edges for mean.
    bin_ampl_from_zero : bool, optional
        Whether to start binning load amplitudes from 0 or from min_amp
        if ampl_bins is an integer. The default is False (bin from min_amp).

    Returns
    -------
    cycles : array
        Array containing the number of cycles of each bin (ampl_bins, mean_bins).
    ampl_bin_mean : array 
        Array containing the mean of the cycle amplitudes of each bin (ampl_bins, mean_bins).
    ampl_edges : array
        Array containing the amplitude bin edges (ampl_bins + 1,).
    mean_bin_mean : array
        Array containing the mean of the cycle means of each bin (ampl_bins, mean_bins).
    mean_edges : array
        Array containing the mean bin edges (mean_bins + 1,).

    '''
    # Rainflow counting
    if isinstance(signals[0], tuple):
        weights, ampls, means = np.array([(np.ones_like(ampl) * weight, ampl, mean) for weight, signal in signals
                                          for ampl, mean in rainflow_func(signal[:]).T], dtype=np.float64).T
    else:
        ampls, means = rainflow_func(signals[:])
        weights = np.ones_like(ampls)
    
    # Bin load cycles by amplitude and mean
    if isinstance(ampl_bins, int) and bin_ampl_from_zero:
        ampl_bins = np.linspace(0, 1, num=ampl_bins + 1) * ampls[weights > 0].max()
    cycles, ampl_edges, mean_edges = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ampl_bin_sum = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights * ampls)[0]
        ampl_bin_mean = ampl_bin_sum / np.where(cycles, cycles, 1)
        mean_bin_sum = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights * means)[0]
        mean_bin_mean = mean_bin_sum / np.where(cycles, cycles, 1)
    cycles = cycles / 2  # to get full cycles
    return cycles, ampl_bin_mean, ampl_edges, mean_bin_mean, mean_edges
