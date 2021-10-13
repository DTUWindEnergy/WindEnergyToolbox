'''
Created on 27/11/2015

@author: MMPE
'''


import numpy as np
import warnings


def spectrum(x, y=None, k=1):
    """PSD or Cross spectrum (only positive half)

    If input time series are two dimensional, then columns are interpreted
    as different time series and the mean spectrum is returned

    Parameters
    ----------
    x : array_like
        Time series
    y : array_like, optional
        If array_like, cross spectrum of x and y is returned
        if None, cross spectrum of x and x (i.e. PSD) is returned
    k : int or float, optional
        Max wave number
    """
    if x is None:
        return None

    def fft(x):
        return np.fft.fft(x.T).T / len(x)

    if y is None or x is y:
        fftx = fft(x)
        fftx = fftx * np.conj(fftx)  # PSD, ~ fft(x)**2
    else:
        fftx = fft(x) * np.conj(fft(y))  # Cross spectra

    fftx = fftx[:len(fftx) // 2] * 2  # positive half * 2

#    if len(fftx.shape) == 2:
#        fftx = np.mean(fftx, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.real(fftx * len(x) / (2 * k))[1:]


def spectra(spatial_resolution, u, v=None, w=None, detrend=True):
    """Return the wave number, the uu, vv, ww autospectra and the uw cross spectra

    Parameters
    ----------
    spatial_resolution : int, float or array_like
        Distance between samples in meters
        - For turbulence boxes: 1/dx = Nx/Lx where dx is distance between points,
        Nx is number of points and Lx is box length in meters
        - For time series: Sample frequency / U
    u : array_like
        The u-wind component\n
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n
    v : array_like, optional
        The v-wind component
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n
    w : array_like, optional
        The w-wind component
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n
    detrend : boolean, optional
        if True (default) wind speeds are detrended before calculating spectra

    Returns
    -------
    k1, uu[, vv[, ww, uw]] : 2-5 x array_like
        - k1: wave number
        - uu: The uu auto spectrum\n
        - vv: The vv auto spectrum, only if v is provided\n
        - ww: The ww auto spectrum, only if w is provided\n
        - uw: The uw cross spectrum, only if w is provided\n

        For two dimensional input, the mean spectra are returned
    """
    assert isinstance(spatial_resolution, (int, float))

    k = 2 * np.pi * spatial_resolution
    if v is not None:
        assert u.shape == v.shape
    if w is not None:
        assert u.shape == w.shape

    if 1 and len(u.shape) == 2:
        #         assert np.abs(np.mean(u, 0)).max() < 2
        #         if v is not None:
        #             assert np.abs(np.mean(v, 0)).max() < 1
        #         if w is not None:
        #             assert np.abs(np.mean(w, 0)).max() < 1
        if isinstance(k, float):
            k = np.repeat(k, u.shape[1])
        else:
            assert u.shape[1] == k.shape[0]
        k1_vec = np.array([np.linspace(0, k_ / 2, len(u) // 2)[1:] for k_ in k]).T
    else:
        #assert np.abs(np.mean(u)) < 1
        if v is not None:
            assert np.abs(np.mean(v)) < 1
        if w is not None:
            assert np.abs(np.mean(w)) < 1
        assert isinstance(k, float)
        k1_vec = np.linspace(0, k / 2, int(len(u) / 2))[1:]
    if detrend:
        u, v, w = detrend_wsp(u, v, w)

    return [k1_vec] + [spectrum(x1, x2, k=k) for x1, x2 in [(u, u), (v, v), (w, w), (w, u)]]


def spectra_from_time_series(sample_frq, Uvw_lst):
    """Return the wave number, the uu, vv, ww autospectra and the uw cross spectra

    Parameters
    ----------
    sample_frq : int, float or array_like
        Sample frequency
    Uvw_lst : array_like
        list of U, v and w, [(U1,v1,w1),(U2,v2,w2)...], v and w are optional

    Returns
    -------
    k1, uu[, vv[, ww, uw]] : 2-5 x array_like
        - k1: wave number
        - uu: The uu auto spectrum\n
        - vv: The vv auto spectrum, only if v is provided\n
        - ww: The ww auto spectrum, only if w is provided\n
        - uw: The uw cross spectrum, only if w is provided\n

        For two dimensional input, the mean spectra are returned
    """
    assert isinstance(sample_frq, (int, float))
    Uvw_arr = np.array(Uvw_lst)
    Ncomp = Uvw_arr.shape[1]
    U, v, w = [Uvw_arr[:, i, :].T for i in range(Ncomp)] + [None] * (3 - Ncomp)
    k = 2 * np.pi * sample_frq / U.mean(0)

    if v is not None:
        assert np.abs(np.nanmean(v, 0)).max() < 1, "Max absolute mean of v is %f" % np.abs(np.nanmean(v, 0)).max()
    if w is not None:
        assert np.abs(np.nanmean(w, 0)).max() < 1
    k1_vec = np.array([np.linspace(0, k_ / 2, U.shape[0] // 2)[1:] for k_ in k]).T
    u = U - np.nanmean(U, 0)
    u, v, w = detrend_wsp(u, v, w)

    return [k1_vec] + [spectrum(x1, x2, k=k) for x1, x2 in [(u, u), (v, v), (w, w), (w, u)]]


def bin_spectrum(x, y, bin_size, min_bin_count=2):
    assert min_bin_count > 0
    x = x / bin_size
    low, high = np.floor(np.nanmin(x)), np.ceil(np.nanmax(x))
    bins = int(high - low)
    nbr_in_bins = np.histogram(x, bins, range=(low, high))[0]
    if len(x.shape) == 2:
        min_bin_count *= x.shape[1]
    mask = nbr_in_bins >= min_bin_count
    return np.histogram(x, bins, range=(low, high), weights=y)[0][mask] / nbr_in_bins[mask], nbr_in_bins


def logbin_spectrum(k1, xx, log10_bin_size=.2, min_bin_count=2):
    ln_bin_size = np.log(10) * log10_bin_size
    if xx is None:
        return None
    return (bin_spectrum(np.log(k1), (xx), ln_bin_size, min_bin_count)[0])


def logbin_spectra(k1, uu, vv=None, ww=None, uw=None, log10_bin_size=0.2, min_bin_count=2):
    return tuple([logbin_spectrum(k1, xx, log10_bin_size, min_bin_count) for xx in [k1, uu, vv, ww, uw]])


def plot_spectrum(spacial_frq, u, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt
    k1, uu = logbin_spectra(*spectra(spacial_frq, u))[:2]

    plt.semilogx(k1, k1 * uu, 'b-')


def detrend_wsp(u, v=None, w=None):
    def _detrend(wsp):
        if wsp is None:
            return None
        dwsp = np.atleast_2d(wsp.copy().T).T
        t = np.arange(dwsp.shape[0])
        A = np.vstack([t, np.ones(len(t))]).T
        for i in range(dwsp.shape[1]):
            m = ~np.isnan(dwsp[:, i])
            trend, offset = np.linalg.lstsq(A[:m.sum()], dwsp[:, i][m], rcond=None)[0]
            dwsp[:, i] = dwsp[:, i] - t * trend + t[-1] / 2 * trend
        return dwsp.reshape(wsp.shape)
    return [_detrend(wsp) for wsp in [u, v, w]]


def plot_spectra(k1, uu, vv=None, ww=None, uw=None, mean_u=1, log10_bin_size=.2, plt=None, marker_style='.'):
    if plt is None:
        import matplotlib.pyplot as plt
    bk1, buu, bvv, bww, buw = logbin_spectra(k1, uu, vv, ww, uw, log10_bin_size)

    def plot(xx, label, color, plt):
        plt.semilogx(bk1, bk1 * xx * 10 ** 0 / mean_u ** 2, marker_style + color, label=label)
    plot(buu, 'uu', 'r', plt)
    plt.xlabel('Wavenumber $k_{1}$ [$m^{-1}$]')
    if mean_u == 1:
        plt.ylabel(r'Spectral density $k_{1} F(k_{1}) [m^2/s^2]$')
    else:
        plt.ylabel(r'Spectral density $k_{1} F(k_{1})/U^{2} [-]$')
    if (bvv) is not None:
        plot(bvv, 'vv', 'g', plt)
    if bww is not None:
        plot(bww, 'ww', 'b', plt)
        plot(buw, 'uw', 'm', plt)
