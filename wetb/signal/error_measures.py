'''
Created on 30/06/2016

@author: MMPE
'''

from scipy.interpolate.interpolate import interp1d

import numpy as np
from wetb.signal.fit import bin_fit


def rms(a, b):
    """Calculate the Root-Mean-Squared Error of two value sets

    Parameters
    ---------
    a : array_like
        First value set
    b : array_like
        Second value set

    Returns
    -------
    y : float
        Root mean squared error of a and b

    """
    a, b = [np.array(ab[:]) for ab in [a, b]]
    if a.shape != b.shape:
        raise ValueError("Dimensions differ: %s!=%s" % (a.shape, b.shape))
    if len(a) == 0:
        return np.nan
    return np.sqrt(np.nanmean((a - b) ** 2))


def rms2fit(x, y, fit_func=bin_fit):
    """
    Calculate the rms error of the points (xi, yi) relative to the mean curve

    The mean curve is computed by:\n
    - Divide x into bins + 1 bins\n
    - Remove bins with less than 2 elements\n
    - Calculate the mean of x and y in the bins\n
    - Do a linear interpolation between the bin mean values\n
    - Extrapolate to the minimum and maximum value of x using the slope of the first and last line segment\n

    Usefull for calculating e.g. power curve scatter

    Parameters
    ---------
    x : array_like
        x values
    y : array_like
        y values
    bins : int or array_like, optional
        If int: Number of control points for the mean curve, default is 10\n
        If array_like: Bin egdes
    kind : str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear',
        'quadratic','cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation
        of first, second or third order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'cubic'.
    fit_func : function, optional
        Function to apply on each bin to find control points for fit
    
    Returns
    -------
    err : float
        Mean error of points compared to mean curve
    f : function
        Interpolation function
    """
    x, y = np.array(x[:]), np.array(y[:])
    _, fit = fit_func(x,y)
    return rms(fit(x),y), fit
    
def rms2fit_old(x, y, bins=10, kind='cubic', fit_func=np.nanmean, normalize_with_slope=False):
    """
    Calculate the rms error of the points (xi, yi) relative to the mean curve

    The mean curve is computed by:\n
    - Divide x into bins + 1 bins\n
    - Remove bins with less than 2 elements\n
    - Calculate the mean of x and y in the bins\n
    - Do a linear interpolation between the bin mean values\n
    - Extrapolate to the minimum and maximum value of x using the slope of the first and last line segment\n

    Usefull for calculating e.g. power curve scatter

    Parameters
    ---------
    x : array_like
        x values
    y : array_like
        y values
    bins : int or array_like, optional
        If int: Number of control points for the mean curve, default is 10\n
        If array_like: Bin egdes
    kind : str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear',
        'quadratic','cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation
        of first, second or third order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'cubic'.
    fit_func : function, optional
        Function to apply on each bin to find control points for fit
    normalize_with_slope : boolean, optional
        If True, the mean error in each bin is normalized with the slope of the corresponding line segment

    Returns
    -------
    err : float
        Mean error of points compared to mean curve
    f : function
        Interpolation function
    """
    x, y = np.array(x[:]), np.array(y[:])
    if isinstance(bins, int):
        bins = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-10, bins + 1)

    digitized = np.digitize(x, bins)
    digitized[np.isnan(x) | np.isnan(y)] = -1

    masks = [digitized == i for i in range(1, len(bins))]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_x = np.array([np.nanmean(x[mask]) for mask in masks])
        bin_y = np.array([fit_func(y[mask]) for mask in masks])
        bin_count = np.array([np.sum(mask) for mask in masks])
    bin_x_fit, bin_y = [b[bin_count >= 1] for b in [bin_x, bin_y]]

    #extrapolate to first and last value of x
    if bin_x_fit[0] > np.nanmin(x):
        bin_y = np.r_[bin_y[0] - (bin_x_fit[0] - np.nanmin(x)) * (bin_y[1] - bin_y[0]) / (bin_x_fit[1] - bin_x_fit[0]), bin_y]
        bin_x_fit = np.r_[np.nanmin(x), bin_x_fit]

    if bin_x_fit[-1] < np.nanmax(x):
        bin_y = np.r_[bin_y, bin_y[-1] + (np.nanmax(x) - bin_x_fit[-1]) * (bin_y[-1] - bin_y[-2]) / (bin_x_fit[-1] - bin_x_fit[-2]) ]
        bin_x_fit = np.r_[bin_x_fit, np.nanmax(x)]

    #Create mean function
    f = lambda x : interp1d(bin_x_fit, bin_y, kind)(x[:])

    #calculate error of segment
    digitized = np.digitize(x, bin_x[bin_count > 0])
    bin_err = np.array([rms(y[digitized == i], f(x[digitized == i])) for i in range(1, len(bin_x_fit))])
    if normalize_with_slope:
        slopes = np.diff(bin_y) / np.diff(bin_x_fit)
        return np.nanmean(bin_err / np.abs(slopes)), f
    return np.sqrt(np.nanmean(bin_err ** 2)), f


def rms2mean(x, y, bins=10, kind='cubic', normalize_with_slope=False):
    """
    Calculate the rms error of the points (xi, yi) relative to the mean curve

    The mean curve is computed by:\n
    - Divide x into bins + 1 bins\n
    - Remove bins with less than 2 elements\n
    - Calculate the mean of x and y in the bins\n
    - Do a linear interpolation between the bin mean values\n
    - Extrapolate to the minimum and maximum value of x using the slope of the first and last line segment\n

    Usefull for calculating e.g. power curve scatter

    Parameters
    ---------
    x : array_like
        x values
    y : array_like
        y values
    bins : int or array_like, optional
        If int: Number of control points for the mean curve, default is 10\n
        If array_like: Bin egdes
    kind : str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear',
        'quadratic','cubic' where 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation
        of first, second or third order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'cubic'.
    normalize_with_slope : boolean, optional
        If True, the mean error in each bin is normalized with the slope of the corresponding line segment

    Returns
    -------
    err : float
        Mean error of points compared to mean curve
    f : function
        Interpolation function
    """
    return rms2fit(x, y, lambda x,y : bin_fit(x, y, bins, kind))


def bootstrap_comparison(x, y, kind=1, N=15, M=100):
    f_lst = []
    y_lst = []
    x_min, x_max = max(np.percentile(x, 2), np.sort(x)[2]), min(np.percentile(x, 98), np.sort(x)[-2])

    y_arr = np.empty((M, N * 10)) + np.NaN
    inside = 0
    for i in range(M):
        indexes = np.random.randint(0, len(x) - 1, len(x))
        while x[indexes].min() > x_min or x[indexes].max() < x_max:
            indexes = np.random.randint(0, len(x) - 1, len(x))
        #indexes = np.arange(i, len(x), M)

        _, f = rms2fit(x[indexes], y[indexes], lambda x,y : bin_fit(x,y, kind=kind, bins=N))
        x_ = np.linspace(x_min, x_max, N * 10)
        y_ = (f(x_))
        if i > 10:
            if np.all(y_ < np.nanmax(y_arr, 0)) and np.all(y_ > np.nanmin(y_arr, 0)):
                inside += 1
                if inside == 5:
                    #print ("break", i)
                    #break
                    pass
        y_arr[i, :] = y_

    return (np.mean(np.std(y_arr, 0)), x_, y_arr[~np.isnan(y_arr[:, 0])])
