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
        stat_data, info = gtsdf.load_statistic(fn)
        self.assertEqual(data[:, 0].min(), stat_data.values[0, 0])
        self.assertEqual(stat_data.shape, (49, 10))

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

        df, info = collect_statistics('.', tmp_path, filename='*stat.hdf5')
        assert df.shape == (98, 12)
        df, info = collect_statistics('.', tmp_path + "..", filename='*stat.hdf5')
        assert df.shape == (98, 12)
        with pytest.raises(Exception, match=r'No \*stat\.hdf5 files found in'):
            collect_statistics('.', tmp_path + "..", filename='*stat.hdf5', recursive=False)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
