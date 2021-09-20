
import numpy as np
from scipy.interpolate.interpolate import interp1d


def bin_fit(x, y, bins=10, kind='linear', bin_func=np.nanmean, bin_min_count=3, lower_upper='discard'):
    """Fit observations based on bin statistics

    Parameters
    ---------
    x : array_like
        x observations
    y : array_like
        y observations
    bins : int, array_like or (int, int)
        if int: <bins> binx evenly distributed on the x-axis
        if (xbins,ybins): <xbins> and <ybins> evenly distributed on the x and y axis respectively\n
        Note that ybins only make sense if every y-value maps to a single x-value
    kind : int or string
        degree of polynomial for fit (argument passed to scipy.interpolate.interpolate.interp1d)
    bin_func : function, optional
        Statistic function to apply on bins, default is nanmean
    bin_min_count : int, optional
        Minimum number of observations in bins to include
        Default is 3
    lower_upper : str, int, (str,str), (int,int)
        How to handle observations below and above first and last bin values. Can be:\n
        - "discard":
        - "extrapolate":
        - int: Set f(max(x)) to mean of first/last int observations


    Returns
    -------
    bin_x, fit_function

    """
    x, y = np.array(x[:]), np.array(y[:])
    if isinstance(bins, int):
        bins = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-10, bins + 1)
    elif isinstance(bins, tuple) and len(bins) == 2 and isinstance(bins[0], int) and isinstance(bins[1], int):
        xbins, ybins = bins
        if xbins > 0:
            xbinsx = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-10, xbins + 1)
        else:
            xbinsx = []
        if ybins > 0:
            x1, f1 = bin_fit(y, x, kind=1, bins=ybins)
            xbinsy = f1(x1)
        else:
            xbinsy = []
        #x2, f2 = bin_fit(x,y, kind=1, bins=xbins)
        bins = sorted(np.r_[xbinsx, xbinsy])

    digitized = np.digitize(x, bins)
    digitized[np.isnan(x) | np.isnan(y)] = -1

    masks = [digitized == i for i in range(1, len(bins))]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_x = np.array([np.nanmean(x[mask]) for mask in masks])
        bin_y = np.array([bin_func(y[mask]) for mask in masks])
        bin_count = np.array([np.sum(mask) for mask in masks])
    #bin_x_fit, bin_y = [b[bin_count >= bin_min_count] for b in [bin_x, bin_y]]
    bin_x_fit = bin_x
    m = np.isnan(bin_x_fit)
    bin_x_fit[m] = ((bins[:-1] + bins[1:]) / 2)[m]
    bin_y_fit = bin_y.copy()
    bin_y_fit[bin_count < bin_min_count] = np.nan

    if isinstance(lower_upper, (str, int)):
        lower = upper = lower_upper
    else:
        lower, upper = lower_upper

    # Add value to min(x)
    if bin_x_fit[0] > np.nanmin(x) or np.isnan(bin_y_fit[0]):
        if lower == 'extrapolate':

            bin_y_fit = np.r_[bin_y_fit[0] - (bin_x_fit[0] - np.nanmin(x)) *
                              (bin_y_fit[1] - bin_y_fit[0]) / (bin_x_fit[1] - bin_x_fit[0]), bin_y_fit]
            bin_x_fit = np.r_[np.nanmin(x), bin_x_fit]
        elif lower == "discard":
            pass
        elif isinstance(lower, int):
            bin_y_fit = np.r_[np.mean(y[~np.isnan(x)][np.argsort(x[~np.isnan(x)])[:lower]]), bin_y_fit]
            bin_x_fit = np.r_[np.nanmin(x), bin_x_fit]
        else:
            raise NotImplementedError("Argument for handling lower observations, %s, not implemented" % lower)

    # add value to max(x)
    if bin_x_fit[-1] < np.nanmax(x) or np.isnan(bin_y_fit[-1]):
        if upper == 'extrapolate':
            bin_y_fit = np.r_[bin_y_fit, bin_y_fit[-1] + (np.nanmax(x) - bin_x_fit[-1])
                              * (bin_y_fit[-1] - bin_y_fit[-2]) / (bin_x_fit[-1] - bin_x_fit[-2])]
            bin_x_fit = np.r_[bin_x_fit, np.nanmax(x)]
        elif upper == "discard":
            pass
        elif isinstance(upper, int):
            bin_y_fit = np.r_[bin_y_fit, np.mean(y[~np.isnan(x)][np.argsort(x[~np.isnan(x)])[-upper:]])]
            bin_x_fit = np.r_[bin_x_fit, np.nanmax(x)]
        else:
            raise NotImplementedError("Argument for handling upper observations, %s, not implemented" % upper)

    return bin_x_fit, _interpolate_fit(bin_x_fit, bin_y_fit, kind)


def perpendicular_bin_fit(x, y, bins=30, fit_func=None, bin_min_count=3, plt=None):
    """Fit a curve to the values, (x,y) using bins that are perpendicular to an initial fit

    Parameters
    ---------
    x : array_like
        x observations
    y : array_like
        y observations
    bins : int
        Number of perpendicular bins
    fit_func : function(x,y) -> (x,y) or None
        Initial fit function
        If None, bin_fit with same number of bins are used
    bin_min_count : int, optional
        Minimum number of observations in bins to include
        Default is 3

    plt : pyplot or None
        If pyplot the fitting process is plotted on plt

    Returns
    -------
    fit_x, fit_y
    """

    if fit_func is None:
        def fit_func(x, y): return bin_fit(x, y, bins, bin_func=np.nanmean)

    x, y = [v[~np.isnan(x) & ~np.isnan(y)] for v in [x, y]]

    bfx, f = fit_func(x, y)
    bfy = f(bfx)

    bfx, bfy = [v[~np.isnan(bfx) & ~np.isnan(bfy)] for v in [bfx, bfy]]
    if plt:
        x_range, y_range = [v.max() - v.min() for v in [x, y]]
        plt.ylim([y.min() - y_range * .1, y.max() + y_range * .1])
        plt.xlim([x.min() - x_range * .1, x.max() + x_range * .1])

    # divide curve into N segments of same normalized curve length
    xg, xo = np.nanmax(bfx) - np.nanmin(bfx), np.nanmin(bfx)
    yg, yo = np.nanmax(bfy) - np.nanmin(bfy), np.nanmin(bfy)
    nbfx = (bfx - xo) / xg
    nbfy = (bfy - yo) / yg
    l = np.cumsum(np.sqrt(np.diff(nbfx)**2 + np.diff(nbfy)**2))
    nx, ny = [np.interp(np.linspace(l[0], l[-1], bins + 1), l, (xy[1:] + xy[:-1]) / 2) for xy in [nbfx, nbfy]]

    last = (-1, 0)

    pc = []
    used = np.zeros_like(x).astype(np.bool)
    for i in range(0, len(nx)):
        i1, i2 = max(0, i - 1), min(len(nx) - 1, i + 1)
        a = -(nx[i2] - nx[i1]) / (ny[i2] - ny[i1])
        b = (ny[i] - (a * nx[i])) * yg + yo
        a *= yg / xg
        x_ = [np.nanmin(x), np.nanmax(x)]
        m1 = np.sign(last[0]) * y < np.sign(last[0]) * ((x - xo) * last[0] + last[1])
        m2 = np.sign(a) * y > np.sign(a) * (a * (x - xo) + b)
        m = m1 & m2 & ~used
        if plt:
            plt.plot(x_, ((a) * (x_ - xo)) + b)
            plt.plot(x[m], y[m], '.')

        if np.sum(m) >= bin_min_count:
            pc.append((np.median(x[m]), np.median(y[m])))
            used = used | m
        last = (a, b)
    #bfx,bfy = zip(*pc)
    if plt:
        pbfx, pbfy = np.array(pc).T
        plt.plot(bfx, bfy, 'orange', label='initial_fit')
        plt.plot(pbfx, pbfy, 'gray', label="perpendicular fit")
        plt.legend()
    #PlotData(None, bfx,bfy)
    bin_x_fit, bin_y_fit = np.array(pc).T
    return bin_x_fit, _interpolate_fit(bin_x_fit, bin_y_fit, kind="linear")

# Create mean function


def _interpolate_fit(bin_x_fit, bin_y_fit, kind='linear'):
    def fit(x):
        x = np.atleast_1d(x)[:].copy().astype(float)
        x[x < bin_x_fit[0]] = np.nan
        x[x > bin_x_fit[-1]] = np.nan
        m = ~(np.isnan(bin_x_fit) | np.isnan(bin_y_fit))
        return interp1d(bin_x_fit[m], bin_y_fit[m], kind)(x[:])
    return fit
