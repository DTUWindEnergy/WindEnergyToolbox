'''
Created on 13/07/2016

@author: MMPE
'''
import numpy as np
from wetb.signal.filters import replacer


replace_by_nan = replacer.replace_by_nan
replace_by_line = replacer.replace_by_line
replace_by_mean = replacer.replace_by_mean
replace_by_polynomial = replacer.replace_by_polynomial





def nanmedian(x):
    """Median ignoring nan (similar to numpy.nanmax"""
    return np.median(x[~np.isnan(x)])


def thresshold_finder(data, thresshold, plt=None):
    spike_mask = np.abs(data) > thresshold
    if plt:
        plt.plot(data, label='Fluctuations')
        plt.plot ([0, data.shape[0]], [thresshold, thresshold], 'r', label='Thresshold')
        plt.plot ([0, data.shape[0]], [-thresshold, -thresshold], 'r')

    return spike_mask

def univeral_thresshold_finder(data, variation='mad', plt=None):

    ## Three variation measures in decreasing order of sensitivity to outliers
    variation = {'std': np.nanstd(data),  # standard deviation
                 'abs': np.nanmean(np.abs(data - np.nanmean(data))),  # mean abs deviation
                 'mad': nanmedian(np.abs(data - nanmedian(data)))  # median abs deviation (mad)
                 }.get(variation, variation)

    thresshold = np.sqrt(2 * np.log(data.shape[0])) * variation  # Universal thresshold (expected maximum of n random variables)
    return thresshold_finder(data, thresshold, plt)

def despike(data, spike_length, spike_finder=univeral_thresshold_finder, spike_replacer=replace_by_nan, plt=None):
    """Despike data

    Parameters
    ---------
    data : array_like
        data
    spike_length : int
        Typical spike duration [samples]
    spike_finder : function
        Function returning indexes of the spikes
    spike_replacer : function
        function that replaces spikes

    Returns
    -------
    Despiked data
    """
    if plt:
        plt.plot(data, label='Input')
    data = np.array(data).copy()
    from wetb.signal.filters import frq_filters
    lp_data = frq_filters.low_pass(data, spike_length, 1)
    hp_data = data - lp_data
    hp_data = frq_filters.high_pass(data, spike_length*4, 1, order=2)
        #spike_length, sample_frq/2, 1, order=1)
    spike_mask = spike_finder(hp_data, plt=plt)
    despike_data = spike_replacer(data, spike_mask)
    if plt:
        plt.plot(despike_data, 'k.', label='Output')
        plt.legend()
    return despike_data

if __name__ == '__main__':
    from wetb import gtsdf
    time, data, info = gtsdf.load(r'C:\data\SWT36Hovsore\metdata/const_10hz.hdf5')
    wsp = data[:, 45]

    import matplotlib.pyplot as plt
    despike(wsp, .1, lambda data, plt : univeral_thresshold_finder(data, 'std', plt), plt=plt)
    plt.show()
