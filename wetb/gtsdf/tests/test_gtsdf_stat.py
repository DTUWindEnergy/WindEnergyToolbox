'''
Created on 12/09/2013

@author: mmpe
'''

import h5py
import numpy as np
from wetb import gtsdf

import unittest
import os
from wetb.gtsdf.gtsdf import add_postproc, load_postproc, collect_postproc, compress2postproc
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
        # test_gtsdf_postproc
        time, data, info = gtsdf.load(tfp + 'test.hdf5')
        fn = tmp_path + "test_stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        add_postproc(fn)
        da = load_postproc(fn)
        sensor = da[0]
        self.assertEqual(data[:, 0].min(), sensor[0].sel(statistic='min'))
        self.assertEqual(da[0].shape, (49, 4))

        # test_gtsdf_compress2postproc
        time, data, info = gtsdf.load(tfp + 'test.hdf5')
        fn = tmp_path + "test_compress2stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        del info['dtype']
        gtsdf.save(tmp_path + "test_compress2stat2.hdf5", data, time=time, dtype=float, **info)
        compress2postproc(fn)
        self.assertLess(os.path.getsize(fn) * 30, os.path.getsize(tfp + 'test.hdf5'))

        # test_collect_postproc
        with pytest.raises(Exception, match=r'No \*\.hdf5 files found in'):
            collect_postproc('missing', tmp_path)

        da = collect_postproc(tmp_path)
        assert da[0].shape == (3, 49, 4)
        da = collect_postproc(tmp_path + "..")
        assert da[0].shape == (4, 49, 4)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
