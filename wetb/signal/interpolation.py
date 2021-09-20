
import numpy as np
import numpy.ma as ma
#from pylab import *
def interpolate(x, xp, yp, max_xp_step=None, max_dydxp=None, cyclic_range=None, max_repeated=None):
    """Interpolation similar to numpy.interp that handles nan and missing values

    Parameters
    ----------
    x : array_like
        The x-coordinates of the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing.
    yp : 1-D sequence of floats
        The y-coordinates of the data points, same length as xp.
    max_xp_step : int, float or None, optional
        Maximum xp-time step that is interpolated to x.\n
        If time step > max_xp_step then NAN is returned for intermediate x values
        If None, default, then this fix is not applied
    max_dydxp : int, float, None, optional
        Maximum absolute dydxp (slop of yp) that is interpolated to y.\n
        If dydxp > max_dydxp then NAN is returned for intermediate y values
        If None, default, then this fix is not applied
    cyclick_range : int, float, None, optional
        Range of posible values, e.g. 360 for degrees (both 0..360 and -180..180)
        If None (default), data not interpreted as cyclic
    max_repeated : int, float, None, optional
        Maximum xp that yp are allowed to be repeated
        if yp[i]==yp[i+1]==..==yp[i+j] and xp[i+j]-xp[i]>max_repeated_yp then
        NAN is returned for xp[i]<x<=xp[i+j]


    Returns
    -------
    y : {float, ndarray}
        The interpolated values, same shape as x.

    Examples
    --------
    >>> interpolate(x=[1,1.5,2,3], xp=[0.5,1.5,3], yp=[5,15,30])
    [10,15,20,30]
    >>> interpolate(x=[0, 1, 2, 7, 12], xp=[0, 2, 12], yp=[359, 0, 350], max_dydxp=45)
    [359,nan,0,175,350]
    """

    xp = np.array(xp, dtype=float)
    yp = np.array(yp, dtype=float)
    assert xp.shape[0] == yp.shape[0], "xp and yp must have same length (%d!=%d)" % (xp.shape[0], yp.shape[0])
    non_nan = ~(np.isnan(xp) & np.isnan(yp))
    yp = yp[non_nan]
    xp = xp[non_nan]
    y = np.interp(x, xp, yp, np.nan, np.nan)


    if cyclic_range is not None:
        cr = cyclic_range
        y2 = np.interp(x, xp, (yp + cr / 2) % cr - cr / 2, np.nan, np.nan) % cr
        y = np.choose(np.r_[0, np.abs(np.diff(y)) > np.abs(np.diff(y2))], np.array([y, y2]))

    if max_xp_step:
        diff = np.diff(xp)
        diff[np.isnan(diff)] = 0

        indexes = (np.where(diff > max_xp_step)[0])
        for index in indexes:
            y[(x > xp[index]) & (x < xp[index + 1])] = np.nan
    if max_dydxp:
        if cyclic_range is None:
            abs_dydxp = np.abs(np.diff(yp) / np.diff(xp))
        else:
            abs_dydxp = np.min([np.abs(np.diff((yp + cyclic_range / 2) % cyclic_range)) , np.abs(np.diff(yp % cyclic_range)) ], 0) / np.diff(xp)
        abs_dydxp[np.isnan(abs_dydxp)] = 0

        indexes = (np.where(abs_dydxp > max_dydxp)[0])
        for index in indexes:
            y[(x > xp[index]) & (x < xp[index + 1])] = np.nan
    if max_repeated:
        rep = np.r_[False, yp[1:] == yp[:-1], False]
        tr = rep[1:] ^ rep[:-1]
        itr = np.where(tr)[0]
        for start, stop, l in zip (itr[::2] , itr[1::2], xp[itr[1::2]] - xp[itr[::2]]):
            if l >= max_repeated:
                y[(x > xp[start]) & (x <= xp[stop])] = np.nan
    return y
#print (interpolate(x=[0, 1, 2, 3, 4], xp=[0, 1, 2, 4], yp=[5, 5, 6, 6], max_repeated=1))
