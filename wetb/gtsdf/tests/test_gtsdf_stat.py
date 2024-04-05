'''
Created on 12/09/2013

@author: mmpe
'''

import h5py
import numpy as np
from wetb import gtsdf

import unittest
import os
from wetb.gtsdf.gtsdf import collect_statistics
import pytest

tmp_path = os.path.dirname(__file__) + "/tmp/"
tfp = os.path.dirname(os.path.abspath(__file__)) + "/test_files/"


class Test_gsdf(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)

    @classmethod
    def tearDownClass(cls):
        super(Test_gsdf, cls).tearDownClass()
        # shutil.rmtree(tmp_path)

    def test_gtsdf_stat(self):
        # test_gtsdf_stat
        time, data, info = gtsdf.load(tfp + 'test.hdf5')
        fn = tmp_path + "test_stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        gtsdf.add_statistic(fn)
        da = gtsdf.load_statistic(fn)
        sensor = da[0]
        self.assertEqual(data[:, 0].min(), sensor.sel(stat='min'))
        assert sensor.sensor_name == info['attribute_names'][0]
        assert sensor.sensor_unit == info['attribute_units'][0]
        assert sensor.sensor_description == info['attribute_descriptions'][0]
        self.assertEqual(da.shape, (49, 10))

        # test_gtsdf_compress2stat
        time, data, info = gtsdf.load(tfp + 'test.hdf5')
        fn = tmp_path + "test_compress2stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        del info['dtype']
        gtsdf.save(tmp_path + "test_compress2stat2.hdf5", data, time=time, dtype=float, **info)
        gtsdf.compress2statistics(fn)
        self.assertLess(os.path.getsize(fn) * 50, os.path.getsize(tfp + 'test.hdf5'))

        # test_collect_stat
        with pytest.raises(Exception, match=r'No \*\.hdf5 files found in'):
            collect_statistics('missing', tmp_path)

        da = collect_statistics('.', tmp_path, filename='*stat.hdf5')
        assert da.shape == (2, 49, 10)
        da = collect_statistics('.', tmp_path + "..", filename='*stat.hdf5')
        assert da.shape == (2, 49, 10)
        with pytest.raises(Exception, match=r'No \*stat\.hdf5 files found in'):
            collect_statistics('.', tmp_path + "..", filename='*stat.hdf5', recursive=False)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
