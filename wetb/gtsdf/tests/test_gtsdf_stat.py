'''
Created on 12/09/2013

@author: mmpe
'''
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
standard_library.install_aliases()

import h5py
import numpy as np
from wetb import gtsdf

import unittest
import os

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
        #shutil.rmtree(tmp_path)

    
    def test_gtsdf_stat(self):
        time, data, info = gtsdf.load(tfp+'test.hdf5')
        fn = tmp_path + "test_stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        gtsdf.add_statistic(fn)
        stat_data,info = gtsdf.load_statistic(fn)
        self.assertEqual(data[:,0].min(), stat_data.values[0,0])
        self.assertEqual(stat_data.shape, (49,10))        
        
    def test_gtsdf_compress2stat(self):
        time, data, info = gtsdf.load(tfp+'test.hdf5')
        fn = tmp_path + "test_compress2stat.hdf5"
        gtsdf.save(fn, data, time=time, **info)
        gtsdf.save(tmp_path + "test_compress2stat2.hdf5", data, time=time, dtype=np.float, **info)
        gtsdf.compress2statistics(fn)
        self.assertLess(os.path.getsize(fn)*50, os.path.getsize(tfp+'test.hdf5'))
        
        






if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
