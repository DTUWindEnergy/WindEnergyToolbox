'''
Created on 16/07/2013

@author: mmpe
'''
import sys

import unittest

import numpy as np
from wetb.fatigue_tools.fatigue import (eq_load, rainflow_astm,
                                        rainflow_windap, cycle_matrix)
from wetb.hawc2 import Hawc2io
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


class TestFatigueTools(unittest.TestCase):

    def test_leq_1hz(self):
        """Simple test of wetb.fatigue_tools.fatigue.eq_load using a sine
        signal.
        """
        amplitude = 1
        m = 1
        point_per_deg = 100

        for amplitude in [1, 2, 3]:
            peak2peak = amplitude * 2
            # sine signal with 10 periods (20 peaks)
            nr_periods = 10
            time = np.linspace(0, nr_periods * 2 * np.pi, point_per_deg * 180)
            neq = time[-1]
            # mean value of the signal shouldn't matter
            signal = amplitude * np.sin(time) + 5
            r_eq_1hz = eq_load(signal, no_bins=1, m=m, neq=neq)[0]
            r_eq_1hz_expected = ((2 * nr_periods * amplitude**m) / neq)**(1 / m)
            np.testing.assert_allclose(r_eq_1hz, r_eq_1hz_expected)

            # sine signal with 20 periods (40 peaks)
            nr_periods = 20
            time = np.linspace(0, nr_periods * 2 * np.pi, point_per_deg * 180)
            neq = time[-1]
            # mean value of the signal shouldn't matter
            signal = amplitude * np.sin(time) + 9
            r_eq_1hz2 = eq_load(signal, no_bins=1, m=m, neq=neq)[0]
            r_eq_1hz_expected2 = ((2 * nr_periods * amplitude**m) / neq)**(1 / m)
            np.testing.assert_allclose(r_eq_1hz2, r_eq_1hz_expected2)

            # 1hz equivalent should be independent of the length of the signal
            np.testing.assert_allclose(r_eq_1hz, r_eq_1hz2)

    def test_rainflow_combi(self):
        """Signal with two frequencies and amplitudes
        """

        amplitude = 1
        # peak2peak = amplitude * 2
        m = 1
        point_per_deg = 100

        nr_periods = 10
        time = np.linspace(0, nr_periods * 2 * np.pi, point_per_deg * 180)

        signal = (amplitude * np.sin(time)) + 5 + (amplitude * 0.2 * np.cos(5 * time))
        cycles, ampl_bin_mean, ampl_edges, mean_bin_mean, mean_edges = \
            cycle_matrix(signal, ampl_bins=10, mean_bins=5)

        cycles.sum()

    def test_astm1(self):

        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])

        ampl, mean = rainflow_astm(signal)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[0., 1., 0., 0.],
                                                                                       [1., 0., 0., 2.],
                                                                                       [0., 0., 0., 0.],
                                                                                       [0., 0., 0., 1.],
                                                                                       [0., 0., 0., 0.],
                                                                                       [0., 0., 1., 2.]]))

    def test_windap1(self):
        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
        ampl, mean = rainflow_windap(signal, 18, 2)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[0., 0., 1., 0.],
                                                                                       [1., 0., 0., 2.],
                                                                                       [0., 0., 0., 0.],
                                                                                       [0., 0., 0., 1.],
                                                                                       [0., 0., 0., 0.],
                                                                                       [0., 0., 2., 1.]]))

    def test_windap2(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61), np.array([[1.356, 1.758, 2.370, 2.784, 3.077, 3.296]]), 0.01)

    def test_astm2(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61, rainflow_func=rainflow_astm),
                                   np.array([[1.356, 1.758, 2.370, 2.784, 3.077, 3.296]]), 0.01)


#     def test_windap3(self):
#         data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
#         from wetb.fatigue_tools.rainflowcounting import peak_trough
#         self.assertTrue(peak_trough.__file__.lower()[-4:] == ".pyd" or peak_trough.__file__.lower()[-3:] == ".so",
#                         "not compiled, %s, %s\n%s"%(sys.executable, peak_trough.__file__, os.listdir(os.path.dirname(peak_trough.__file__))))
#         np.testing.assert_array_equal(cycle_matrix(data, 4, 4, rainflow_func=rainflow_windap)[0], np.array([[  14., 65., 39., 24.],
#                                                                    [  0., 1., 4., 0.],
#                                                                    [  0., 0., 0., 0.],
#                                                                    [  0., 1., 2., 0.]]) / 2)

    def test_astm3(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(cycle_matrix(data, 4, 4, rainflow_func=rainflow_astm)[0], np.array([[24., 83., 53., 26.],
                                                                                                       [0., 1., 4., 0.],
                                                                                                       [0., 0., 0., 0.],
                                                                                                       [0., 1., 2., 0.]]) / 2, 0.001)

    def test_astm_weighted(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(cycle_matrix([(1, data), (1, data)], 4, 4, rainflow_func=rainflow_astm)[0], np.array([[24., 83., 53., 26.],
                                                                                                                         [0., 1.,
                                                                                                                             4., 0.],
                                                                                                                         [0., 0.,
                                                                                                                             0., 0.],
                                                                                                                         [0., 1., 2., 0.]]), 0.001)

    def test_astm_matlab_example(self):
        # example from https://se.mathworks.com/help/signal/ref/rainflow.html
        fs = 512

        X = np.array([-2, 1, -3, 5, -1, 3, -4, 4, -2])

        Y = -np.diff(X)[:, np.newaxis] / 2. * np.cos(np.pi *
                                                     np.arange(0, 1, 1 / fs))[np.newaxis] + ((X[:-1] + X[1:]) / 2)[:, np.newaxis]
        Y = np.r_[Y.flatten(), X[-1]]
        range_lst, mean_lst = (rainflow_astm(Y))
        np.testing.assert_array_equal(range_lst, [3, 4, 4, 4, 8, 9, 8, 6])
        np.testing.assert_array_equal(mean_lst, [-.5, -1, 1, 1, 1, .5, 0, 1])
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, len(X) - 1 + 1 / fs, 1 / fs), Y)
            plt.plot(np.arange(len(X)), X, 'o')
            plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
