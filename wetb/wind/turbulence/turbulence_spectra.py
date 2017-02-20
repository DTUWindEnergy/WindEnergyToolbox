'''
Created on 27/11/2015

@author: MMPE
'''


import numpy as np
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

    fftx = fftx[:len(fftx) // 2 ] * 2  # positive half * 2

    if len(fftx.shape) == 2:
        fftx = np.mean(fftx, 1)

    return np.real(fftx * len(x) / (2 * k))[1:]

def spectra(spacial_frq, u, v=None, w=None, detrend=True):
    """Return the wave number, the uu, vv, ww autospectra and the uw cross spectra

    Parameters
    ----------
    spacial_frq : inf or float
        - For time series: Sample frequency, see Notes\n
        - For turbulence boxes: 1/dx where
        dx is nx/Lx (number of grid points in length direction / box length in meters)
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

    Notes
    -----
    If the 'mean wind speed' or 'shear compensated reference wind speed' are subtracted from u in advance,
    then the spacial_frq must be divided by the subtracted wind speed
    """
    assert isinstance(spacial_frq, (int, float))

    k = 2 * np.pi * spacial_frq

    if np.mean(u) > 1:
        k /= np.mean(u)
        u = u - np.mean(u, 0)
    if detrend:
        u, v, w = detrend_wsp(u, v, w)

    k1_vec = np.linspace(0, k / 2, len(u) / 2)[1:]
    return [k1_vec] + [spectrum(x1, x2, k=k) for x1, x2 in [(u, u), (v, v), (w, w), (w, u)]]


def bin_spectrum(x, y, bin_size, min_bin_count=2):
    assert min_bin_count > 0
    x = x / bin_size
    low, high = np.floor(np.min(x)), np.ceil(np.max(x))
    bins = int(high - low)
    nbr_in_bins = np.histogram(x, bins, range=(low, high))[0]
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
            trend, offset = np.linalg.lstsq(A, dwsp[:, i])[0]
            dwsp[:, i] = dwsp[:, i] - t * trend + t[-1] / 2 * trend
        return dwsp.reshape(wsp.shape)
    return [_detrend(wsp) for wsp in [u, v, w]]
