'''
Created on 03/06/2014

@author: MMPE
'''


import os

from scipy.interpolate import RectBivariateSpline

import numpy as np
from wetb.wind.turbulence.turbulence_spectra import spectra, logbin_spectra, \
    plot_spectrum



sp1, sp2, sp3, sp4 = np.load(os.path.dirname(__file__).replace("library.zip", '') + "/mann_spectra_data.npy")

yp = np.arange(-3, 3.1, 0.1)
xp = np.arange(0, 5.1, 0.1)
RBS1 = RectBivariateSpline(xp, yp, sp1)
RBS2 = RectBivariateSpline(xp, yp, sp2)
RBS3 = RectBivariateSpline(xp, yp, sp3)
RBS4 = RectBivariateSpline(xp, yp, sp4)

def estimate_mann_parameters(sf, u, v, w=None):
    if isinstance(u, (list, tuple)):
        #return fit_mann_model_spectra(*mean_spectra(sf, u, v, w))
        raise NotImplementedError
    else:
        return fit_mann_model_spectra(*spectra(sf, u, v, w))

#def mean_spectra(fs, u_ref_lst, u_lst, v_lst=None, w_lst=None):
#    if isinstance(fs, (int, float)):
#        fs = [fs] * len(u_lst)
#    if v_lst is None:
#        u_lst = [spectra(fs, u_ref, u) for fs, u_ref, u in zip(fs, u_ref_lst, u_lst)]
#        return [np.mean(np.array([ku[i] for ku in u_lst]), 0) for i in range(2)]
#    if w_lst is None:
#        uv_lst = [spectra(fs, u_ref, u, v) for fs, u_ref, u, v in zip(fs, u_ref_lst, u_lst, v_lst)]
#        return [np.mean(np.array([kuv[i] for kuv in uv_lst]), 0) for i in range(3)]
#    else:
#        uvw_lst = [spectra(fs, u_ref, u, v, w) for fs, u_ref, u, v, w in zip(fs, u_ref_lst, u_lst, v_lst, w_lst)]
#        return [np.mean(np.array([kuvw[i] for kuvw in uvw_lst]), 0) for i in range(5)]






def get_mann_model_spectra(ae, L, G, k1):
    """Mann model spectra

    Parameters
    ----------
    ae : int or float
        Alpha epsilon^(2/3) of Mann model
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model
    k1 : array_like
        Desired wave numbers

    Returns
    -------
    uu : array_like
        The u-autospectrum of the wave numbers, k1
    vv : array_like
        The v-autospectrum of the wave numbers, k1
    ww : array_like
        The w-autospectrum of the wave numbers, k1
    uw : array_like
        The u,w cross spectrum of the wave numbers, k1
    """
    xq = np.log10(L * k1)
    yq = (np.zeros_like(xq) + G)
    f = L ** (5 / 3) * ae
    uu = f * RBS1.ev(yq, xq)
    vv = f * RBS2.ev(yq, xq)
    ww = f * RBS3.ev(yq, xq)
    uw = f * RBS4.ev(yq, xq)
    return uu, vv, ww, uw




def _local_error(x, k1, uu, vv, ww=None, uw=None):

    ae, L, G = x
    val = 10 ** 99
    if ae >= 0 and G >= 0 and G <= 5 and L > 0 and np.log10(k1[0] * L) >= -3 and np.log10(k1[0] * L) <= 3:
        tmpuu, tmpvv, tmpww, tmpuw = get_mann_model_spectra(ae, L, G, k1)
        val = np.sum((k1 * uu - k1 * tmpuu) ** 2)
        if vv is not None:
            val += np.sum((k1 * vv - k1 * tmpvv) ** 2)
        if ww is not None:
            val += np.sum((k1 * ww - k1 * tmpww) ** 2) + np.sum((k1 * uw - k1 * tmpuw) ** 2)
    return val

def fit_mann_model_spectra(k1, uu, vv=None, ww=None, uw=None, log10_bin_size=.2, min_bin_count=2, start_vals_for_optimisation=(0.01, 50, 3.3), plt=False):
    """Fit a mann model to the spectra

    Bins the spectra, into logarithmic sized bins and find the mann model parameters,
    that minimize the error between the binned spectra and the Mann model spectra
    using an optimization function

    Parameters
    ----------
    k1 : array_like
        Wave numbers
    uu : array_like
        The u-autospectrum of the wave numbers, k1
    vv : array_like, optional
        The v-autospectrum of the wave numbers, k1
    ww : array_like, optional
        The w-autospectrum of the wave numbers, k1
    uw : array_like, optional
        The u,w cross spectrum of the wave numbers, k1
    log10_bin_size : int or float, optional
        Bin size (log 10, based)
    start_vals_for_optimization : (ae, L, G), optional
        - ae: Alpha epsilon^(2/3) of Mann model\n
        - L: Length scale of Mann model\n
        - G: Gamma of Mann model

    Returns
    -------
    ae : int or float
        Alpha epsilon^(2/3) of Mann model
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model

    Examples
    --------
    >>> sf = sample_frq / u_ref
    >>> u,v,w = # u,v,w wind components
    >>> ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v, w))
    >>> u1,v1 = # u,v wind components
    >>> ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v))
    """
    from scipy.optimize import fmin
    x = fmin(_local_error, start_vals_for_optimisation, logbin_spectra(k1, uu, vv, ww, uw, log10_bin_size, min_bin_count), disp=False)

    if plt is not False:
        if not hasattr(plt, 'plot'):
            import matplotlib.pyplot as plt
        _plot_spectra(k1, uu, vv, ww, uw, plt=plt)
        plot_mann_spectra(*x, plt=plt)
        plt.title('ae:%.3f, L:%.1f, G:%.2f' % tuple(x))
        plt.xlabel = ('Wavenumber, k1 [m$^{-1}$]')
        plt.xlabel = ('Spectral density, $k_1F(k1) [m^2/s^2]$')
        plt.legend()
        plt.show()
    return x

def residual(ae, L, G, k1, uu, vv=None, ww=None, uw=None, log10_bin_size=.2):
    """Fit a mann model to the spectra

    Bins the spectra, into logarithmic sized bins and find the mann model parameters,
    that minimize the error between the binned spectra and the Mann model spectra
    using an optimization function

    Parameters
    ----------
    ae : int or float
        Alpha epsilon^(2/3) of Mann model
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model
    k1 : array_like
        Wave numbers
    uu : array_like
        The u-autospectrum of the wave numbers, k1
    vv : array_like, optional
        The v-autospectrum of the wave numbers, k1
    ww : array_like, optional
        The w-autospectrum of the wave numbers, k1
    uw : array_like, optional
        The u,w cross spectrum of the wave numbers, k1
    log10_bin_size : int or float, optional
        Bin size (log 10, based)
    start_vals_for_optimization : (ae, L, G), optional
        - ae: Alpha epsilon^(2/3) of Mann model\n
        - L: Length scale of Mann model\n
        - G: Gamma of Mann model

    Returns
    -------
    residual : float
    """
    _3to2list = list(np.array(logbin_spectra(k1, uu, vv, ww, uw, log10_bin_size)))
    bk1, sp_meas = _3to2list[:1] + [_3to2list[1:]]
    sp_fit = np.array(logbin_spectra(k1, *get_mann_model_spectra(ae, L, G, k1)))[1:]

    return np.sqrt(((bk1 * sp_meas - bk1 * sp_fit) ** 2).mean())

def fit_ae(sf, u, L, G, min_bin_count=None, plt=False):
    """Fit alpha-epsilon to match variance of time series

    Parameters
    ----------
    sf : int or float
        Sample frequency
    u : array-like
        u vind component
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model

    Returns
    -------
    ae : float
        Alpha epsilon^(2/3) of Mann model that makes the energy of the model equal to the varians of u
    """
#    def get_var(k1, ae):
#        uu = get_mann_model_spectra(ae, L, G, k1)[0]
#        return np.trapz(2 * uu, k1)
#        return (np.sum(uu) * 2 * (k1[1] - k1[0]))
    if len(u.shape) == 1:
        u = u.reshape(len(u), 1)
    if min_bin_count is None:
        min_bin_count = max(2, 6 - u.shape[1] / 2)

    def get_var(k1, uu):
        l = 0  #128 // np.sqrt(u.shape[1])
        return np.trapz(2 * uu[l:], k1[l:])

    k1 = spectra(sf, u)[0]
    v1 = get_var(*logbin_spectra(k1, get_mann_model_spectra(0.1, L, G, k1)[0], min_bin_count=min_bin_count)[:2])
    v2 = get_var(*logbin_spectra(k1, get_mann_model_spectra(0.2, L, G, k1)[0], min_bin_count=min_bin_count)[:2])
    k1, uu = logbin_spectra(*spectra(sf, u), min_bin_count=2)[:2]
    #variance = np.mean([detrend_wsp(u_)[0].var() for u_ in u.T])
    v = get_var(k1, uu)
    ae = (v - v1) / (v2 - v1) * .1 + .1
    if plt is not False:
        if not hasattr(plt, 'plot'):
            import matplotlib.pyplot as plt
        plt.semilogx(k1, k1 * uu, 'b-', label='uu')
        k1, uu = logbin_spectra(*spectra(sf, u), min_bin_count=1)[:2]

        plt.semilogx(k1, k1 * uu, 'r--', label='uu_logbin')
        muu = get_mann_model_spectra(ae, L, G, k1)[0]
        plt.semilogx(k1, k1 * muu, 'g', label='ae:%.3f, L:%.1f, G:%.2f' % (ae, L, G))
        plt.legend()
        plt.xlabel = ('Wavenumber, k1 [m$^{-1}$]')
        plt.xlabel = ('Spectral density, $k_1F(k1) [m^2/s^2]$')
        plt.show()
    return ae


def plot_spectra(ae, L, G, k1, uu, vv, ww=None, uw=None, mean_u=1, log10_bin_size=.2, show=True, plt=None):
#    if plt is None:
#        import matplotlib.pyplot as plt
    _plot_spectra(k1, uu, vv, ww, uw, mean_u, log10_bin_size, plt)
    plot_mann_spectra(ae, L, G, "-", mean_u, plt)
    if show:
        plt.show()

def _plot_spectra(k1, uu, vv, ww=None, uw=None, mean_u=1, log10_bin_size=.2, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt
    bk1, buu, bvv, bww, buw = logbin_spectra(k1, uu, vv, ww, uw, log10_bin_size)
    def plot(xx, label, color, plt):
        plt.semilogx(bk1, bk1 * xx * 10 ** 0 / mean_u ** 2 , '.' + color)
    plot(buu, 'uu', 'r', plt)
    if (bvv) is not None:
        plot(bvv, 'vv', 'g', plt)
    if bww is not None:
        plot(bww, 'ww', 'b', plt)
        plot(buw, 'uw', 'm', plt)

def plot_mann_spectra(ae, L, G, style='-', u_ref=1, plt=None, spectra=['uu', 'vv', 'ww', 'uw']):
    if plt is None:
        import matplotlib.pyplot as plt
    mf = 10 ** (np.linspace(-4, 1, 1000))
    muu, mvv, mww, muw = get_mann_model_spectra(ae, L, G, mf)
    if 'uu' in spectra: plt.semilogx(mf, mf * muu * 10 ** 0 / u_ref ** 2, 'r' + style)
    if 'vv' in spectra:     plt.semilogx(mf, mf * mvv * 10 ** 0 / u_ref ** 2, 'g' + style)
    if 'ww' in spectra: plt.semilogx(mf, mf * mww * 10 ** 0 / u_ref ** 2, 'b' + style)
    if 'uw' in spectra: plt.semilogx(mf, mf * muw * 10 ** 0 / u_ref ** 2, 'm' + style)



if __name__ == "__main__":
    from wetb import gtsdf
    from wetb.wind.dir_mapping import wsp_dir2uv
    
    """Example of fitting Mann parameters to a time series"""
    from wetb import wind
    ds = gtsdf.Dataset(os.path.dirname(wind.__file__)+"/tests/test_files/WspDataset.hdf5")#'unit_test/test_files/wspdataset.hdf5')
    f = 35
    u, v = wsp_dir2uv(ds.Vhub_85m, ds.Dir_hub_)

    u_ref = np.mean(u)
    u -= u_ref

    sf = f / u_ref
    import matplotlib.pyplot as plt
    ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v), plt = plt)
    print (ae, L, G)

 
    """Example of fitting Mann parameters to a "series" of a turbulence box"""
    l = 16384
    nx = 8192
    ny, nz = 8, 8
    sf = (nx / l)
    fn = os.path.dirname(wind.__file__)+"/tests/test_files/turb/h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat"
    u, v, w = [np.fromfile(fn % uvw, np.dtype('<f'), -1).reshape(nx , ny * nz) for uvw in ['u', 'v', 'w']]
    ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v, w), plt=plt)
    print (ae, L, G)
