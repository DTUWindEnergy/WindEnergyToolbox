'''
Created on 16/07/2013

@author: mmpe
'''
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
import sys
standard_library.install_aliases()

import unittest

import numpy as np
from wetb.fatigue_tools.fatigue import eq_load, rainflow_astm, rainflow_windap, \
    cycle_matrix
from wetb.hawc2 import Hawc2io
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path

class TestFatigueTools(unittest.TestCase):


    def test_astm1(self):

        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])

        ampl, mean = rainflow_astm(signal)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[ 0., 1., 0., 0.],
                                                                                                           [ 1., 0., 0., 2.],
                                                                                                           [ 0., 0., 0., 0.],
                                                                                                           [ 0., 0., 0., 1.],
                                                                                                           [ 0., 0., 0., 0.],
                                                                                                           [ 0., 0., 1., 2.]]))

    def test_windap1(self):
        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
        ampl, mean = rainflow_windap(signal, 18, 2)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[ 0., 0., 1., 0.],
                                                                                       [ 1., 0., 0., 2.],
                                                                                       [ 0., 0., 0., 0.],
                                                                                       [ 0., 0., 0., 1.],
                                                                                       [ 0., 0., 0., 0.],
                                                                                       [ 0., 0., 2., 1.]]))

    def test_windap2(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61), np.array([[1.356, 1.758, 2.370, 2.784, 3.077, 3.296]]), 0.01)


    def test_astm2(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61, rainflow_func=rainflow_astm), np.array([[1.356, 1.758, 2.370, 2.784, 3.077, 3.296]]), 0.01)


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
        np.testing.assert_allclose(cycle_matrix(data, 4, 4, rainflow_func=rainflow_astm)[0], np.array([[ 24., 83., 53., 26.],
                                                                                                           [  0., 1., 4., 0.],
                                                                                                           [  0., 0., 0., 0.],
                                                                                                           [  0., 1., 2., 0.]]) / 2, 0.001)
        
    def test_astm_weighted(self):
        data = Hawc2io.ReadHawc2(testfilepath + "test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(cycle_matrix([(1, data),(1,data)], 4, 4, rainflow_func=rainflow_astm)[0], np.array([[ 24., 83., 53., 26.],
                                                                                                           [  0., 1., 4., 0.],
                                                                                                           [  0., 0., 0., 0.],
                                                                                                           [  0., 1., 2., 0.]]) , 0.001)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
