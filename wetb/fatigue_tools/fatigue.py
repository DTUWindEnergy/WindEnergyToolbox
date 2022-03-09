'''
Created on 04/03/2013
@author: mmpe


'eq_load' calculate equivalent loads using one of the two rain flow counting methods
'cycle_matrix' calculates a matrix of cycles (binned on amplitude and mean value)
'eq_load_and_cycles' is used to calculate eq_loads of multiple time series (e.g. life time equivalent load)

The methods uses the rainflow counting routines (See documentation in top of methods):
- 'rainflow_windap': (Described in "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads",
                      2. edition 1990, Appendix A)
or
- 'rainflow_astm' (based on the c-implementation by Adam Nieslony found at the MATLAB Central File Exchange
                   http://www.mathworks.com/matlabcentral/fileexchange/3026)
'''
import warnings
import numpy as np
from wetb.fatigue_tools.rainflowcounting import rainflowcount

rainflow_windap = rainflowcount.rainflow_windap
rainflow_astm = rainflowcount.rainflow_astm


def eq_load(signals, no_bins=46, m=[3, 4, 6, 8, 10, 12], neq=1, rainflow_func=rainflow_windap):
    """Equivalent load calculation

    Calculate the equivalent loads for a list of Wohler exponent and number of equivalent loads

    Parameters
    ----------
    signals : list of tuples or array_like
        - if list of tuples: list must have format [(sig1_weight, sig1),(sig2_weight, sig1),...] where\n
            - sigx_weight is the weight of signal x\n
            - sigx is signal x\n
        - if array_like: The signal
    no_bins : int, optional
        Number of bins in rainflow count histogram
    m : int, float or array-like, optional
        Wohler exponent (default is [3, 4, 6, 8, 10, 12])
    neq : int, float or array-like, optional
        The equivalent number of load cycles (default is 1, but normally the time duration in seconds is used)
    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap)

    Returns
    -------
    eq_loads : array-like
        List of lists of equivalent loads for the corresponding equivalent number(s) and Wohler exponents

    Examples
    --------
    >>> signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
    >>> eq_load(signal, no_bins=50, neq=[1, 17], m=[3, 4, 6], rainflow_func=rainflow_windap)
    [[10.311095426959747, 9.5942535021382174, 9.0789213365013932], # neq = 1, m=[3,4,6]
    [4.010099657859783, 4.7249689509841746, 5.6618639965313005]], # neq = 17, m=[3,4,6]

    eq_load([(.4, signal), (.6, signal)], no_bins=50, neq=[1, 17], m=[3, 4, 6], rainflow_func=rainflow_windap)
    [[10.311095426959747, 9.5942535021382174, 9.0789213365013932], # neq = 1, m=[3,4,6]
    [4.010099657859783, 4.7249689509841746, 5.6618639965313005]], # neq = 17, m=[3,4,6]
    """
    try:
        return eq_load_and_cycles(signals, no_bins, m, neq, rainflow_func)[0]
    except TypeError:
        return [[np.nan] * len(np.atleast_1d(m))] * len(np.atleast_1d(neq))


def eq_load_and_cycles(signals, no_bins=46, m=[3, 4, 6, 8, 10, 12], neq=[10 ** 6, 10 ** 7, 10 ** 8], rainflow_func=rainflow_windap):
    """Calculate combined fatigue equivalent load

    Parameters
    ----------
    signals : list of tuples or array_like
        - if list of tuples: list must have format [(sig1_weight, sig1),(sig2_weight, sig1),...] where\n
            - sigx_weight is the weight of signal x\n
            - sigx is signal x\n
        - if array_like: The signal
    no_bins : int, optional
        Number of bins for rainflow counting
    m : int, float or array-like, optional
        Wohler exponent (default is [3, 4, 6, 8, 10, 12])
    neq : int or array-like, optional
        Equivalent number, default is [10^6, 10^7, 10^8]
    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap)

    Returns
    -------
    eq_loads : array-like
        List of lists of equivalent loads for the corresponding equivalent number(s) and Wohler exponents
    cycles : array_like
        2d array with shape = (no_ampl_bins, 1)
    ampl_bin_mean : array_like
        mean amplitude of the bins
    ampl_bin_edges
        Edges of the amplitude bins
    """
    cycles, ampl_bin_mean, ampl_bin_edges, _, _ = cycle_matrix(signals, no_bins, 1, rainflow_func)
    if 0:  #to be similar to windap
        ampl_bin_mean = (ampl_bin_edges[:-1] + ampl_bin_edges[1:]) / 2
    cycles, ampl_bin_mean = cycles.flatten(), ampl_bin_mean.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq_loads = [[((np.nansum(cycles * ampl_bin_mean ** _m) / _neq) ** (1. / _m)) for _m in np.atleast_1d(m)]  for _neq in np.atleast_1d(neq)]
    return eq_loads, cycles, ampl_bin_mean, ampl_bin_edges


def cycle_matrix(signals, ampl_bins=10, mean_bins=10, rainflow_func=rainflow_windap):
    """Markow load cycle matrix

    Calculate the Markow load cycle matrix

    Parameters
    ----------
    Signals : array-like or list of tuples
        - if array-like, the raw signal\n
        - if list of tuples, list of (weight, signal), e.g. [(0.1,sig1), (0.8,sig2), (.1,sig3)]\n
    ampl_bins : int or array-like, optional
        if int, Number of amplitude value bins (default is 10)
        if array-like, the bin edges for amplitude
    mean_bins : int or array-like, optional
        if int, Number of mean value bins (default is 10)
        if array-like, the bin edges for mea
    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap)

    Returns
    -------
    cycles : ndarray, shape(ampl_bins, mean_bins)
        A bi-dimensional histogram of load cycles(full cycles). Amplitudes are\
        histogrammed along the first dimension and mean values are histogrammed along the second dimension.
    ampl_bin_mean : ndarray, shape(ampl_bins,)
        The average cycle amplitude of the bins
    ampl_edges : ndarray, shape(ampl_bins+1,)
        The amplitude bin edges
    mean_bin_mean : ndarray, shape(ampl_bins,)
        The average cycle mean of the bins
    mean_edges : ndarray, shape(mean_bins+1,)
        The mean bin edges

    Examples
    --------
    >>> signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
    >>> cycles, ampl_bin_mean, ampl_edges, mean_bin_mean, mean_edges = cycle_matrix(signal)
    >>> cycles, ampl_bin_mean, ampl_edges, mean_bin_mean, mean_edges = cycle_matrix([(.4, signal), (.6,signal)])
    """

    if isinstance(signals[0], tuple):
        weights, ampls, means = np.array([(np.zeros_like(ampl)+weight,ampl,mean) for weight, signal in signals for ampl,mean in rainflow_func(signal[:]).T], dtype=np.float64).T
    else:
        ampls, means = rainflow_func(signals[:])
        weights = np.ones_like(ampls)
    if isinstance(ampl_bins, int):
        ampl_bins = np.linspace(0, 1, num=ampl_bins + 1) * ampls[weights>0].max()
    cycles, ampl_edges, mean_edges = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ampl_bin_sum = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights * ampls)[0]
        ampl_bin_mean = np.nanmean(ampl_bin_sum / np.where(cycles,cycles,np.nan),1)
        mean_bin_sum = np.histogram2d(ampls, means, [ampl_bins, mean_bins], weights=weights * means)[0]
        mean_bin_mean = np.nanmean(mean_bin_sum / np.where(cycles, cycles, np.nan), 1)
    cycles = cycles / 2  # to get full cycles
    return cycles, ampl_bin_mean, ampl_edges, mean_bin_mean, mean_edges


def cycle_matrix2(signal, nrb_amp, nrb_mean, rainflow_func=rainflow_windap):
    """
    Same as wetb.fatigue_tools.fatigue.cycle_matrix but bin from min_amp to
    max_amp instead of 0 to max_amp.

    Parameters
    ----------

    Signal : ndarray(n)
        1D Raw signal array

    nrb_amp : int
        Number of bins for the amplitudes

    nrb_mean : int
        Number of bins for the means

    rainflow_func : {rainflow_windap, rainflow_astm}, optional
        The rainflow counting function to use (default is rainflow_windap)

    Returns
    -------

    cycles : ndarray, shape(ampl_bins, mean_bins)
        A bi-dimensional histogram of load cycles(full cycles). Amplitudes are\
        histogrammed along the first dimension and mean values are histogrammed
        along the second dimension.

    ampl_edges : ndarray, shape(no_bins+1,n)
        The amplitude bin edges

    mean_edges : ndarray, shape(no_bins+1,n)
        The mean bin edges

    """
    bins = [nrb_amp, nrb_mean]
    ampls, means = rainflow_func(signal)
    weights = np.ones_like(ampls)
    cycles, ampl_edges, mean_edges = np.histogram2d(ampls, means, bins,
                                                    weights=weights)
    cycles = cycles / 2  # to get full cycles

    return cycles, ampl_edges, mean_edges


if __name__ == "__main__":
    signal1 = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
    signal2 = signal1 * 1.1

    # equivalent load for default wohler slopes
    print (eq_load(signal1, no_bins=50, neq=17, rainflow_func=rainflow_windap))
    print (eq_load(signal1, no_bins=50, neq=17, rainflow_func=rainflow_astm))

    # Cycle matrix with 4 amplitude bins and 4 mean value bins
    print (cycle_matrix(signal1, 4, 4, rainflow_func=rainflow_windap))
    print (cycle_matrix(signal1, 4, 4, rainflow_func=rainflow_astm))

    # Cycle matrix where signal1 and signal2 contributes with 50% each
    print (cycle_matrix([(.5, signal1), (.5, signal2)], 4, 8, rainflow_func=rainflow_astm))

