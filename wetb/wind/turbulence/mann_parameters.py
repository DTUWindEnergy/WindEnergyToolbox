'''
Created on 03/06/2014

@author: MMPE
'''


import os

from scipy.interpolate import RectBivariateSpline

import numpy as np
from wetb.wind.turbulence.spectra import spectra, logbin_spectra, plot_spectra,\
    detrend_wsp


sp1, sp2, sp3, sp4 = np.load(os.path.dirname(__file__).replace("library.zip", '') + "/mann_spectra_data.npy")

yp = np.arange(-3, 3.1, 0.1)
xp = np.arange(0, 5.1, 0.1)
RBS1 = RectBivariateSpline(xp, yp, sp1)
RBS2 = RectBivariateSpline(xp, yp, sp2)
RBS3 = RectBivariateSpline(xp, yp, sp3)
RBS4 = RectBivariateSpline(xp, yp, sp4)


# def mean_spectra(fs, u_ref_lst, u_lst, v_lst=None, w_lst=None):
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
    m = (yq >= 0) & (yq <= 5) & (xq >= -3) & (xq <= 3)
    uu, vv, ww, uw = [f * np.where(m, RBS.ev(yq, xq), np.nan) for RBS in [RBS1, RBS2, RBS3, RBS4]]
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


def fit_mann_model_spectra(k1, uu, vv=None, ww=None, uw=None, log10_bin_size=.2,
                           min_bin_count=2, start_vals_for_optimisation=(0.01, 50, 3.3), plt=False):
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
    x = fmin(_local_error, start_vals_for_optimisation, logbin_spectra(
        k1, uu, vv, ww, uw, log10_bin_size, min_bin_count), disp=False)

    if plt:
        if not hasattr(plt, 'plot'):
            import matplotlib.pyplot as plt
#         plot_spectra(k1, uu, vv, ww, uw, plt=plt)
#         plot_mann_spectra(*x, plt=plt)
        ae, L, G = x
        plot_fit(ae, L, G, k1, uu, vv, ww, uw, log10_bin_size=log10_bin_size, plt=plt)
        plt.title('ae:%.3f, L:%.1f, G:%.2f' % tuple(x))
        plt.xlabel('Wavenumber $k_{1}$ [$m^{-1}$]')
        plt.ylabel(r'Spectral density $k_{1} F(k_{1})/U^{2} [m^2/s^2]$')
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
    residual : array_like
        rms of each spectrum
    """
    k1_sp = np.array([sp for sp in logbin_spectra(k1, uu, vv, ww, uw, log10_bin_size) if sp is not None])
    bk1, sp_meas = k1_sp[0], k1_sp[1:]
    sp_fit = np.array(get_mann_model_spectra(ae, L, G, bk1))[:sp_meas.shape[0]]
    return np.sqrt(((bk1 * (sp_meas - sp_fit)) ** 2).mean(1))


def var2ae(variance, L, G, U, T=600, sample_frq=10, plt=False):
    """Fit alpha-epsilon to match variance of time series

    Parameters
    ----------
    variance : array-like
        variance of u vind component
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model
    U : int or float
        Mean wind speed
    T: int or float
        Length [s] of signal, from which the variance is calculated
    sample_frq: int or float
        Sample frequency [Hz] of signal from which the variance is calculated

    Returns
    -------
    ae : float
        Alpha epsilon^(2/3) of Mann model that makes the energy of the model in the
        frequency range [1/length, sample_frq] equal to the variance of u
    """

    k_low, k_high = 2 * np.pi / (U * np.array([T, 1 / sample_frq]))
    k1 = 10 ** (np.linspace(np.log10(k_low), np.log10(k_high), 1000))

    def get_var(uu):
        return np.trapz(2 * uu[:], k1[:])

    v1 = get_var(get_mann_model_spectra(0.1, L, G, k1)[0])
    v2 = get_var(get_mann_model_spectra(0.2, L, G, k1)[0])
    ae = (variance - v1) / (v2 - v1) * .1 + .1
    if plt is not False:
        if not hasattr(plt, 'plot'):
            import matplotlib.pyplot as plt
        muu = get_mann_model_spectra(ae, L, G, k1)[0]
        plt.semilogx(k1, k1 * muu, label='ae:%.3f, L:%.1f, G:%.2f' % (ae, L, G))
        plt.legend()
        plt.xlabel('Wavenumber $k_{1}$ [$m^{-1}$]')
        plt.ylabel(r'Spectral density $k_{1} F(k_{1})/U^{2} [m^2/s^2]$')
    return ae


def ae2ti(ae23, L, G, U, T=600, sample_frq=10):
    k_low, k_high = 2 * np.pi / (U * np.array([T, 1 / sample_frq]))
    k1 = 10 ** (np.linspace(np.log10(k_low), np.log10(k_high), 1000))

    uu = get_mann_model_spectra(ae23, L, G, k1)[0]
    var = np.trapz(2 * uu[:], k1[:])
    return np.sqrt(var) / U


def fit_ae(spatial_resolution, u, L, G, plt=False):
    """Fit alpha-epsilon to match variance of time series

    Parameters
    ----------
    spatial_resolution : int, float or array_like
        Number of points pr meterDistance between samples in meters
        - For turbulence boxes: 1/dx = Nx/Lx where dx is distance between points,
        Nx is number of points and Lx is box length in meters
        - For time series: Sample frequency / U
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
    # if len(u.shape) == 1:
    #    u = u.reshape(len(u), 1)
#     if min_bin_count is None:
#         min_bin_count = max(2, 6 - u.shape[0] / 2)
#     min_bin_count = 1
    def get_var(k1, uu):
        l = 0  # 128 // np.sqrt(u.shape[1])
        return np.mean(np.trapz(2 * uu[l:], k1[l:], axis=0))

    k1, uu = spectra(spatial_resolution, u)[:2]
    v = get_var(k1, uu)
    v1 = get_var(k1, get_mann_model_spectra(0.1, L, G, k1)[0])
    v2 = get_var(k1, get_mann_model_spectra(0.2, L, G, k1)[0])
    ae = (v - v1) / (v2 - v1) * .1 + .1
#     print (ae)
#
#     k1 = spectra(sf, u)[0]
#     v1 = get_var(*logbin_spectra(k1, get_mann_model_spectra(0.1, L, G, k1)[0], min_bin_count=min_bin_count)[:2])
#     v2 = get_var(*logbin_spectra(k1, get_mann_model_spectra(0.2, L, G, k1)[0], min_bin_count=min_bin_count)[:2])
#     k1, uu = logbin_spectra(*spectra(sf, u), min_bin_count=2)[:2]
#     #variance = np.mean([detrend_wsp(u_)[0].var() for u_ in u.T])
#     v = get_var(k1, uu)
#     ae = (v - v1) / (v2 - v1) * .1 + .1
#     print (ae)
    if plt is not False:
        if not hasattr(plt, 'plot'):
            import matplotlib.pyplot as plt
        plt.semilogx(k1, k1 * uu, 'b-', label='uu')
        k1_lb, uu_lb = logbin_spectra(*spectra(spatial_resolution, u), min_bin_count=1)[:2]

        plt.semilogx(k1_lb, k1_lb * uu_lb, 'r--', label='uu_logbin')
        muu = get_mann_model_spectra(ae, L, G, k1)[0]
        plt.semilogx(k1, k1 * muu, 'g', label='ae:%.3f, L:%.1f, G:%.2f' % (ae, L, G))
        plt.legend()
        plt.xlabel('Wavenumber $k_{1}$ [$m^{-1}$]')
        plt.ylabel(r'Spectral density $k_{1} F(k_{1})/U^{2} [m^2/s^2]$')
        plt.show()
    return ae


def plot_fit(ae, L, G, k1, uu, vv=None, ww=None, uw=None, mean_u=1, log10_bin_size=.2, plt=None):
    #    if plt is None:
    #        import matplotlib.pyplot as plt
    plot_spectra(k1, uu, vv, ww, uw, mean_u, log10_bin_size, plt)
    plot_mann_spectra(ae, L, G, "-", mean_u, plt)


def plot_mann_spectra(ae, L, G, style='-', u_ref=1, plt=None, spectra=['uu', 'vv', 'ww', 'uw']):
    if plt is None:
        import matplotlib.pyplot as plt
    mf = 10 ** (np.linspace(-4, 3, 1000))
    mf = 10 ** (np.linspace(-4, 3, 1000000))
    muu, mvv, mww, muw = get_mann_model_spectra(ae, L, G, mf)
    plt.title("ae: %.3f, L: %.2f, G:%.2f" % (ae, L, G))
    if 'uu' in spectra:
        plt.semilogx(mf, mf * muu * 10 ** 0 / u_ref ** 2, 'r' + style)
    if 'vv' in spectra:
        plt.semilogx(mf, mf * mvv * 10 ** 0 / u_ref ** 2, 'g' + style)
    if 'ww' in spectra:
        plt.semilogx(mf, mf * mww * 10 ** 0 / u_ref ** 2, 'b' + style)
    if 'uw' in spectra:
        plt.semilogx(mf, mf * muw * 10 ** 0 / u_ref ** 2, 'm' + style)


if __name__ == "__main__":
    from wetb import gtsdf
    from wetb.wind.utils import wsp_dir2uv
    from wetb import wind
    import matplotlib.pyplot as plt

    """Example of fitting Mann parameters to a time series"""
    ds = gtsdf.Dataset(os.path.dirname(wind.__file__) +
                       "/tests/test_files/WspDataset.hdf5")  # 'unit_test/test_files/wspdataset.hdf5')
    f = 35
    u, v = wsp_dir2uv(ds.Vhub_85m, ds.Dir_hub_)

    u_ref = np.mean(u)
    u -= u_ref

    sf = f / u_ref
    ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v), plt=None)
    print(ae, L, G)
    print(u.shape)
    print(ds.Time[-1])
    plt.plot(u)
    plt.plot(detrend_wsp(u)[0])
    plt.show()
    print(fit_ae(sf, detrend_wsp(u)[0], L, G, plt))
    print(var2ae(detrend_wsp(u)[0].var(), L, G,))
    print()
    print(fit_ae(sf, u[:21000], L, G))
    print(var2ae(u[:21000].var(), L, G,))


#     """Example of fitting Mann parameters to a "series" of a turbulence box"""
#     l = 16384
#     nx = 8192
#     ny, nz = 8, 8
#     sf = (nx / l)
#     #fn = os.path.dirname(wind.__file__)+"/tests/test_files/turb/h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat"
#     #fn = os.path.dirname(wind.__file__)+"/tests/test_files/turb/h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat"
#     u, v, w = [np.fromfile(fn % uvw, np.dtype('<f'), -1).reshape(nx , ny * nz) for uvw in ['u', 'v', 'w']]
#     ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v, w), plt=plt)
#     print (ae, L, G)


#     """Example of fitting Mann parameters to a "series" of a turbulence box"""
#     l = 4778.3936
#     nx = 8192
#     ny, nz = 32,32
#     sf = (nx / l)
#     #fn = os.path.dirname(wind.__file__)+"/tests/test_files/turb/h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat"
#     for s in [1,2,3,4,5,6]:
#         fn = r'C:\mmpe\HAWC2\models\SWT3.6-107\turb/mann_l73.1_ae0.00_g2.0_h1_8192x32x32_0.583x3.44x3.44_s160%d%%s.turb'%s
#         u, v, w = [np.fromfile(fn % uvw, np.dtype('<f'), -1).reshape(nx , ny * nz) for uvw in ['u', 'v', 'w']]
#         u = np.fromfile(fn % 'u', np.dtype('<f'), -1).reshape(nx , ny * nz)
#         #ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v, w), plt=plt)
#         #print (fit_ae(sf, u, 73.0730383576,  2.01636095317))
#         print (u.std())
#     #print (ae, L, G)
