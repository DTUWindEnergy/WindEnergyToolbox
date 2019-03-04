'''
Created on 30. mar. 2017

@author: mmpe
'''
import numpy as np
from wetb.signal.fit._spline_fit import spline_fit
from wetb.signal.filters._differentiation import differentiation


def fix_rotor_position(rotor_position, sample_frq, rotor_speed, fix_dt=None, plt=None):
    """Rotor position fitted with spline

    Parameters
    ----------
    rotor_position : array_like
        Rotor position [deg] (0-360)
    sample_frq : int or float
        Sample frequency [Hz]
    rotor_speed : array_like
        Rotor speed [RPM]
    fix_dt : int, float or None, optional
        Time distance [s] between spline fix points\n
        If None (default) a range of seconds is tested and the result that minimize the RMS 
        between differentiated rotor position fit and rotor speed is used.\n 
        Note that a significant speed up is achievable by specifying the parameter
    plt : PyPlot or None
        If PyPlot a visual interpretation is plotted

    Returns
    -------
    y : nd_array
        Fitted rotor position
    """

    from wetb.signal.subset_mean import revolution_trigger

    t = np.arange(len(rotor_position))
    indexes = revolution_trigger(rotor_position[:], sample_frq, rotor_speed, max_rev_diff=4)

    rp = rotor_position[:].copy()

    for i in indexes:
        rp[i:] += 360

    if fix_dt is None:
        fix_dt = find_fix_dt(rotor_position, sample_frq, rotor_speed)

    N = int(np.round(fix_dt * sample_frq))
    N2 = N // 2
    if plt:
        a = (rp.max() - rp.min()) / t.max()
        plt.plot(t / sample_frq, rp - t * a, label='Continuous rotor position (detrended)')

    points = []
    for j, i in enumerate(range(0, len(rp), N)):
        # indexes for subsets for overlapping a and b polynomials
        i1 = max(0, i - N2)
        i2 = min(len(rp), i + N2)

        # fit a polynomial
        if i1 < len(rp):
            poly_coef = np.polyfit(t[i1:i2] - t[i1], rp[i1:i2], 1)
            points.append((t[i], np.poly1d(poly_coef)(t[i] - t[i1])))
            if plt:
                plt.plot(t[i1:i2] / sample_frq, np.poly1d(poly_coef)(t[i1:i2] - t[i1]) - t[i1:i2]
                         * a, 'mc'[j % 2], label=('', "Line fit for points (detrended)")[j < 2])

    x, y = np.array(points).T

    if plt:
        plt.plot(x / sample_frq, y - x * a, '.', label='Fit points (detrended)')
        plt.plot(t / sample_frq, spline_fit(x, y)(t) - t * a, label='Spline (detrended)')
        plt.legend()
        plt.show()
    fit = spline_fit(x, y)(t)
    fit[differentiation(fit, "left") < 0] = np.nan
    return fit % 360


def find_fix_dt(rotor_position, sample_frq, rotor_speed, plt=None):
    """Find the optimal fix_dt parameter for fix_rotor_position (function above).
    Optimal is defined as the value that minimizes the sum of squared differences 
    between differentiated rotor position and rotor speed 

    Parameters
    ----------
    rotor_position : array_like
        Rotor position [deg] (0-360)
    sample_frq : int or float
        Sample frequency [Hz]
    rotor_speed : array_like
        Rotor speed [RPM]
    plt : pyplot or None
        If pyplot, a visual interpretation is plotted

    Returns
    -------
    y : int
        Optimal value for the fix_dt parameter for fix_rotor_position

    """
    from wetb.signal.filters import differentiation

    def err(i):
        drp = differentiation(fix_rotor_position(rotor_position, sample_frq, rotor_speed, i))
        rpm_pos = drp % 180 / 360 * sample_frq * 60
        return np.sum((rpm_pos - rotor_speed)**2)

    best = 27
    for step in [9, 3, 1]:

        x_lst = np.arange(-2, 3) * step + best
        res = [err(x) for x in x_lst]
        if plt is not None:
            plt.plot(x_lst, res, '.-')
        best = x_lst[np.argmin(res)]
    if plt is not None:
        plt.show()

    return best


def check_rotor_position(rotor_position, sample_frq, rotor_speed, max_rev_diff=1, plt=None):
    """Rotor position fitted with spline

    Parameters
    ----------
    rotor_position : array_like
        Rotor position [deg] (0-360)
    sample_frq : int or float
        Sample frequency [Hz]
    rotor_speed : array_like
        Rotor speed [RPM]
    fix_dt : int, float or None, optional
        Time distance [s] between spline fix points\n
        If None (default) a range of seconds is tested and the result that minimize the RMS 
        between differentiated rotor position fit and rotor speed is used.\n 
        Note that a significant speed up is achievable by specifying the parameter
    plt : PyPlot or None
        If PyPlot a visual interpretation is plotted

    Returns
    -------
    y : nd_array
        Fitted rotor position
    """

    from wetb.signal.subset_mean import revolution_trigger

    t = np.arange(len(rotor_position)) / sample_frq
    indexes = revolution_trigger(rotor_position[:], sample_frq, rotor_speed, max_rev_diff=max_rev_diff)

    rotor_position = rotor_position[:] % 360

    rp_fit = fix_rotor_position(rotor_position, sample_frq, rotor_speed, None)

    if plt:
        #a = (rp.max()-rp.min())/t.max()
        #plt.plot(t/sample_frq,rp-t*a,label='Continus rotor position (detrended)')
        f, axarr = plt.subplots(2)
        print(rp_fit)
        axarr[0].plot(t, rotor_position, label='rotor position')
        axarr[0].plot(indexes / sample_frq, rotor_position[indexes], '.', label='Revolution trigger')
        axarr[0].plot(t, rp_fit, label='Fitted rotor position')
        axarr[0].legend()
        axarr[1].plot(t, rotor_speed, label='Rotor speed')
        axarr[1].plot(indexes / sample_frq, 60 / (differentiation(indexes) /
                                                  sample_frq), label='Rotor speed (revolution average)')
        print(t[170:200])
        drp = differentiation(rp_fit)
        #drp[(drp<0)]= 0
        axarr[1].plot(t, drp % 180 / 360 * sample_frq * 60, label='Fix(calc from fitted rotor position)')
        axarr[1].legend()

    i1, i2 = indexes[0], indexes[-1]
    print("Rev from rotor position", np.sum(np.diff(rotor_position[i1:i2]) % 180) / 360)
    print("Rev from rotor speed", np.mean(rotor_speed[i1:i2]) * (i2 - i1) / 60 / sample_frq)
    print("Rev from fitted rotor position", np.sum(np.diff(rp_fit[i1:i2]) % 180) / 360)

    print("Mean RPM from rotor speed", np.mean(rotor_speed))
    print("Mean RPM from fitted rotor position", np.sum(
        np.diff(rp_fit[i1:i2]) % 180) / 360 / ((i2 - i1) / 60 / sample_frq))

    spr1 = (np.diff(indexes) / sample_frq)

    #rs1 = 60/( np.diff(indexes)/sample_frq)
    spr2 = np.array([60 / rotor_speed[i1:i2].mean() for i1, i2 in zip(indexes[:-1], indexes[1:])])

    err = spr1 - spr2
    print(err.max())
