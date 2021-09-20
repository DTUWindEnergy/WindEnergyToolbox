'''
Created on 18/07/2016

@author: MMPE
'''
import os
import unittest

import numpy as np
from wetb import gtsdf
from wetb.signal.subset_mean import time_trigger, subset_mean, \
    non_nan_index_trigger, revolution_trigger_old, revolution_trigger
from wetb.utils.geometry import rpm2rads


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')
class TestSubsetMean(unittest.TestCase):
    def test_time_trigger(self):
        time = np.arange(0, 99.5, .5)
        np.testing.assert_array_equal(time[time_trigger(time, 20)], [0, 20, 40, 60, 80])
        np.testing.assert_array_equal(time[time_trigger(time + .5, 20)], [0, 20, 40, 60, 80])
        np.testing.assert_array_equal(time[time_trigger(time + 100000000.5, 20)], [0, 20, 40, 60, 80])
        np.testing.assert_array_equal(time[time_trigger(time, 20, 20, 60)], [20, 40, 60])
        np.testing.assert_array_equal(time_trigger(np.arange(101), 20), [0, 20, 40, 60, 80, 100])
        time, data, info = gtsdf.load(tfp + "subset_mean_test.hdf5")
        np.testing.assert_array_equal(time_trigger(time, 200), [0, 5000, 10000, 15000])

    def test_subset_mean(self):

        time, data, info = gtsdf.load(tfp + "subset_mean_test.hdf5")
        triggers = time_trigger(time, 100)
        t, p = subset_mean([time, data[:, 0]], triggers).T
        self.assertEqual(t[1], time[2500:5000].mean())
        self.assertEqual(p[1], data[2500:5000, 0].mean())

        triggers[1] = 2501
        t, p = subset_mean([time, data[:, 0]], triggers).T
        self.assertEqual(t[1], time[2501:5000].mean())
        self.assertEqual(p[1], data[2501:5000, 0].mean())

    def test_non_nan_index_trigger(self):
        sensor = np.arange(18).astype(float)
        sensor[[5, 11]] = np.nan
        triggers = non_nan_index_trigger(sensor, 3)
        for i1, i2 in triggers:
            self.assertFalse(np.any(np.isnan(sensor[i1:i2])))
        self.assertEqual(len(triggers), 4)


    def test_subset_mean_trigger_tuple(self):
        sensor = np.arange(18).astype(float)
        triggers = non_nan_index_trigger(sensor, 3)
        np.testing.assert_array_equal(subset_mean(sensor, triggers), [ 1., 4., 7., 10., 13., 16])

        #start with nan eq step, eq len
        sensor[0] = np.nan
        triggers = non_nan_index_trigger(sensor, 3)
        np.testing.assert_array_equal(subset_mean(sensor, triggers), [ 2, 5, 8, 11, 14])

        #nan in the middle, noneq step and len
        sensor = np.arange(18).astype(float)
        sensor[[5, 11]] = np.nan
        triggers = non_nan_index_trigger(sensor, 3)

        np.testing.assert_array_equal(subset_mean(sensor, triggers), [1, 7, 13, 16])

    def test_cycle_trigger(self):
        ds = gtsdf.Dataset(tfp+'azi.hdf5')
        azi, rpm, time = [ds(x)[8403:8803] for x in ['azi','Rot_cor','Time']]
        
        trigger = revolution_trigger_old(azi)
        np.testing.assert_array_equal(trigger, [ 17, 128, 241, 354])
        azi[64] = 358
        trigger = revolution_trigger_old(azi, (ds('Rot_cor'), np.diff(time).mean()))
        
#         import matplotlib.pyplot as plt
#         t = np.arange(len(azi))
#         plt.plot(t, azi)
#         for i1,i2 in trigger:
#             plt.plot(t[i1:i2],azi[i1:i2],'.--')
#         plt.show()
        np.testing.assert_array_equal(trigger, [ (128,241),(241,354)])


    def test_revolution_trigger(self):
        rotor_position = np.arange(0.,360*10,4)
        rotor_position += np.random.random(len(rotor_position))
        rotor_position = rotor_position % 360
        if 0:
            x1 = np.random.randint(0, len(rotor_position),10)
            print (list(x1))
            x2 = np.random.randint(0, len(rotor_position),10)
            print (list(x2))
        else:
            x1 = [447, 854,   847, 488, 412, 199, 675]
            x2 = [92, 647, 821, 422, 33, 159, 369, 99, 157, 464]
            
        rotor_position[x1] += 360
        rotor_position[x2] -= 360
        rotor_position[90] = 180
        rotor_position[270:276] = 15
        rotor_position[598:602] = [360,360,0,0]
        rotor_position[748:752] = [360,360,0,0]
        
        indexes = revolution_trigger(rotor_position, 20,1/(90/20/60))
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(rotor_position)
            plt.plot(indexes, np.zeros_like(indexes),'.')
            plt.show()
        np.testing.assert_array_equal(indexes, [ 91, 180, 270, 360, 450, 540, 630, 720, 810])
        
    def test_revolution_trigger_max_rev_diff(self):
        rotor_position = np.arange(0.,360*10,4)
        rotor_position += np.random.random(len(rotor_position))
        rotor_position = rotor_position % 360
        rotor_speed = 1/(90/20/60)
        revolution_trigger(rotor_position, 20, rotor_speed*1.02, max_rev_diff=3)
        self.assertRaisesRegex(AssertionError, "No of revolution mismatch", revolution_trigger, rotor_position, 20, rotor_speed*1.02, max_rev_diff=1)
        
        
if __name__ == "__main__":
    unittest.main()
