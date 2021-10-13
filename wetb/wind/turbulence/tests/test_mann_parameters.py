'''
Created on 05/06/2012

@author: Mads
'''
import os
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from wetb.utils.test_files import get_test_file, move2test_files
from wetb.wind.turbulence.mann_turbulence import load_uvw, fit_mann_parameters,\
    fit_mann_parameters_from_time_series
from wetb.utils.timing import print_time
from wetb.wind.turbulence.spectra import spectra_from_time_series, spectra,\
    plot_spectra, logbin_spectra
import warnings
from wetb.wind.turbulence import mann_turbulence
from wetb.wind.turbulence.mann_parameters import var2ae, fit_ae, ae2ti
from numpy import spacing


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')


class TestMannTurbulence(unittest.TestCase):

    def test_fit_mann_parameters_turbulence_box(self):
        # for uvw in 'uvw':
        #    move2test_files(tfp + 'h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat'%uvw)
        fn_lst = [get_test_file('h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat' % uvw) for uvw in 'uvw']
        u, v, w = load_uvw(fn_lst, (8192, 8, 8))
        dx = 16384 / 8192
        fx = 1 / dx  # spatial resolution
        plt = None
        ae, L, G = fit_mann_parameters(fx, u, v, w, plt=plt)

        self.assertAlmostEqual(ae, .15, delta=0.01)
        self.assertAlmostEqual(L, 10, delta=0.3)
        self.assertAlmostEqual(G, 3.3, delta=0.06)

    def test_spectra_from_timeseries(self):
        fn_lst = [get_test_file('h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat' % uvw) for uvw in 'uvw']
        u, v, w = load_uvw(fn_lst, (8192, 8, 8))
        dx = 16384 / 8192
        fx = 1 / dx  # spatial resolution
        k1, uu, vv, ww, uw = logbin_spectra(*spectra(fx, u, v, w))

        U = u + 4
        sample_frq = 2
        k12, uu2, vv2, ww2, uw2 = logbin_spectra(
            *spectra_from_time_series(sample_frq, [(U_, v_, w_) for U_, v_, w_ in zip(U.T, v.T, w.T)]))
        np.testing.assert_allclose(uu, uu2, 0.02)

        U = u + 8
        sample_frq = 2
        k13, uu3, vv3, ww3, uw3 = logbin_spectra(
            *spectra_from_time_series(sample_frq, [(U_[::2], v_[::2], w_[::2]) for U_, v_, w_ in zip(U.T, v.T, w.T)]))
        np.testing.assert_allclose(uu[:-3], uu3[:-2], 0.1)

        # One set of time series with U=4
        U = u + 4
        Uvw_lst = [(U_, v_, w_) for U_, v_, w_ in zip(U.T, v.T, w.T)]
        # Another set of time series with U=8 i.e. only every second point to have
        # same sample_frq. (nan added to have same length)
        U = u + 4
        Uvw_lst.extend([(np.r_[U_[::2], U_[::2] + np.nan], np.r_[v_[::2], v_[::2] + np.nan],
                         np.r_[w_[::2], w_[::2] + np.nan]) for U_, v_, w_ in zip(U.T, v.T, w.T)])
        sample_frq = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k14, uu4, vv4, ww4, uw4 = logbin_spectra(*spectra_from_time_series(sample_frq, Uvw_lst))
        np.testing.assert_allclose(uu[:-3], uu3[:-2], rtol=0.1)
        if 0:
            import matplotlib.pyplot as plt
            plt.semilogx(k1, uu * k1)
            plt.semilogx(k12, uu2 * k12)
            plt.semilogx(k13, uu3 * k13)
            plt.semilogx(k14, uu4 * k14)
            plt.show()

    def test_fit_mann_parameters_from_timeseries(self):
        fn_lst = [get_test_file('h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat' % uvw) for uvw in 'uvw']
        u, v, w = load_uvw(fn_lst, (8192, 8, 8))
        dx = 16384 / 8192
        fx = 1 / dx  # spatial resolution
        ae, L, G = fit_mann_parameters(fx, u, v, w)
        self.assertAlmostEqual(ae, .15, delta=0.01)
        self.assertAlmostEqual(L, 10, delta=0.3)
        self.assertAlmostEqual(G, 3.3, delta=0.06)

        #import matplotlib.pyplot as plt
        plt = None
        U = u + 4
        sample_frq = 2
        ae, L, G = fit_mann_parameters_from_time_series(
            sample_frq, [(U_, v_, w_) for U_, v_, w_ in zip(U.T, v.T, w.T)], plt=plt)
        self.assertAlmostEqual(ae, .15, delta=0.01)
        self.assertAlmostEqual(L, 10, delta=0.3)
        self.assertAlmostEqual(G, 3.3, delta=0.06)

        # One set of time series with U=4
        U = u + 4
        Uvw_lst = [(U_, v_, w_) for U_, v_, w_ in zip(U.T, v.T, w.T)]
        # Another set of time series with U=8 i.e. only every second point to have
        # same sample_frq. (nan added to have same length)
        U = u + 4
        Uvw_lst.extend([(np.r_[U_[::2], U_[::2] + np.nan], np.r_[v_[::2], v_[::2] + np.nan],
                         np.r_[w_[::2], w_[::2] + np.nan]) for U_, v_, w_ in zip(U.T, v.T, w.T)])
        sample_frq = 2
        ae, L, G = fit_mann_parameters_from_time_series(sample_frq, Uvw_lst, plt=plt)
        self.assertAlmostEqual(ae, .15, delta=0.01)
        self.assertAlmostEqual(L, 10, delta=0.3)
        self.assertAlmostEqual(G, 3.3, delta=0.06)

    def test_var2ae_U(self):
        u = mann_turbulence.load(get_test_file("h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat"), (8192, 8, 8))
        dx = 2
        for U in [1, 10, 100]:
            # should be independent of U
            dt = dx / U
            T = 16384 / U
            self.assertAlmostEqual(var2ae(variance=u.var(), L=10, G=3.3, U=U, T=T, sample_frq=1 / dt), .15, delta=.021)

    def test_var2ae_T(self):
        u = mann_turbulence.load(get_test_file("h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat"), (8192, 8, 8))
        dx = 2
        U = 10
        dt = dx / U
        for i in np.arange(6):
            # reshape to more and shorter series. Variance should decrease while ae should be roughly constant
            n = 2**i
            u_ = u.T.reshape((u.T.shape * np.array([n, 1 / n])).astype(int)).T
            var = u_.var(0).mean()
            ae = var2ae(variance=var, L=10, G=3.3, U=U, T=dx * u_.shape[0] / U, sample_frq=1 / dt)
            self.assertAlmostEqual(ae, .15, delta=.025)

    def test_var2ae_dt(self):
        u = mann_turbulence.load(get_test_file("h2a16384_8_8_65536_32_32_0.15_40_4.0u.dat"), (16384, 8, 8))
        dx = 4
        U = 10
        T = u.shape[0] * dx / U
        for i in np.arange(9):
            # average every neighbouring samples to decrease dt.
            # Variance should decrease while ae should be roughly constant
            n = 2**i

            u_ = u.reshape(u.shape[0] // n, n, u.shape[1]).mean(1)
            var = u_.var(0).mean()

            ae = var2ae(variance=var, L=40, G=4, U=U, T=T, sample_frq=1 / (n * dx / U), plt=False)
            #print(u_.shape, var, ae)
            self.assertAlmostEqual(ae, .15, delta=.04)

    def test_fit_ae2var(self):
        u = mann_turbulence.load(get_test_file("h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat"), (8192, 8, 8))
        self.assertAlmostEqual(fit_ae(spatial_resolution=2, u=u, L=10, G=3.3), .15, delta=.02)

    def test_ae2ti(self):
        u = mann_turbulence.load(get_test_file("h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat"), (8192, 8, 8))
        dx = 2
        U = 10
        dt = dx / U
        T = 16384 / U
        ae23 = var2ae(variance=u.var(), L=10, G=3.3, U=U, T=T, sample_frq=1 / dt)
        ti = u.std() / U
        self.assertAlmostEqual(ae2ti(ae23, L=10, G=3.3, U=U, T=T, sample_frq=1 / dt), ti)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
