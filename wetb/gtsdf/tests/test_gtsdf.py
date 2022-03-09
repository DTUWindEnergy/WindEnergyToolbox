'''
Created on 12/09/2013

@author: mmpe
'''

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
        # shutil.rmtree(tmp_path)

    def test_minimum_requirements(self):
        fn = tmp_path + "minimum.hdf5"
        f = h5py.File(fn, "w")
        # no type
        self.assertRaises(ValueError, gtsdf.load, fn)
        f.attrs["type"] = "General time series data format"

        # no no_blocks
        self.assertRaises(ValueError, gtsdf.load, fn)
        f.attrs["no_blocks"] = 0

        # no block0000
        self.assertRaises(ValueError, gtsdf.load, fn)
        b = f.create_group("block0000")

        # no data
        self.assertRaises(ValueError, gtsdf.load, fn)
        b.create_dataset("data", data=np.empty((0, 0)))
        f.close()
        gtsdf.load(fn)

    def test_save_no_hdf5_ext(self):
        fn = tmp_path + "no_hdf5_ext"
        gtsdf.save(fn, np.arange(12).reshape(4, 3))
        _, _, info = gtsdf.load(fn + ".hdf5")
        self.assertEqual(info['name'], 'no_hdf5_ext')

    def test_load_filename(self):
        fn = tmp_path + "filename.hdf5"
        gtsdf.save(fn, np.arange(12).reshape(4, 3))
        _, _, info = gtsdf.load(fn)
        self.assertEqual(info['name'], 'filename')

    def test_load_fileobject(self):
        fn = tmp_path + "fileobject.hdf5"
        gtsdf.save(fn, np.arange(12).reshape(4, 3))
        _, _, info = gtsdf.load(fn)
        self.assertEqual(info['name'], 'fileobject')

    def test_save_wrong_no_attr_info(self):
        fn = tmp_path + "wrong_no_attr_info.hdf5"
        self.assertRaises(AssertionError, gtsdf.save, fn, np.arange(12).reshape(4, 3), attribute_names=['Att1'])
        self.assertRaises(AssertionError, gtsdf.save, fn, np.arange(12).reshape(4, 3), attribute_units=['s'])
        self.assertRaises(AssertionError, gtsdf.save, fn, np.arange(12).reshape(4, 3), attribute_descriptions=['desc'])

    def test_info(self):
        fn = tmp_path + "info.hdf5"
        gtsdf.save(fn, np.arange(12).reshape(6, 2),
                   name='datasetname',
                   description='datasetdescription',
                   attribute_names=['att1', 'att2'],
                   attribute_units=['s', 'm/s'],
                   attribute_descriptions=['d1', 'd2'])
        _, _, info = gtsdf.load(fn)
        self.assertEqual(info['name'], "datasetname")
        self.assertEqual(info['type'], "General time series data format")
        self.assertEqual(info['description'], "datasetdescription")
        self.assertEqual(list(info['attribute_names']), ['att1', 'att2'])
        self.assertEqual(list(info['attribute_units']), ['s', 'm/s'])
        self.assertEqual(list(info['attribute_descriptions']), ['d1', 'd2'])

    def test_no_time(self):
        fn = tmp_path + 'time.hdf5'
        gtsdf.save(fn, np.arange(12).reshape(6, 2))
        time, _, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(time, np.arange(6))

    def test_int_time(self):
        fn = tmp_path + 'time.hdf5'
        gtsdf.save(fn, np.arange(12).reshape(6, 2), time=range(4, 10))
        time, _, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(time, range(4, 10))

    def test_time_offset(self):
        fn = tmp_path + 'time.hdf5'
        gtsdf.save(fn, np.arange(12).reshape(6, 2), time=range(6), time_start=4)
        time, _, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(time, range(4, 10))

    def test_time_gain_offset(self):
        fn = tmp_path + 'time.hdf5'
        gtsdf.save(fn, np.arange(12).reshape(6, 2), time=range(6), time_step=1 / 4, time_start=4)
        time, _, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(time, np.arange(4, 5.5, .25))

    def test_float_time(self):
        fn = tmp_path + 'time.hdf5'
        gtsdf.save(fn, np.arange(12).reshape(6, 2), time=np.arange(4, 5.5, .25))
        time, _, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(time, np.arange(4, 5.5, .25))

    def test_data(self):
        fn = tmp_path + 'data.hdf5'
        d = np.arange(12).reshape(6, 2)
        gtsdf.save(fn, d)
        f = h5py.File(fn)
        self.assertEqual(f['block0000']['data'].dtype, np.uint16)
        f.close()
        _, data, _ = gtsdf.load(fn)
        np.testing.assert_array_almost_equal(data, np.arange(12).reshape(6, 2), 4)

    def test_data_float(self):
        fn = tmp_path + 'time.hdf5'
        d = np.arange(12).reshape(6, 2)
        gtsdf.save(fn, d, dtype=np.float32)
        f = h5py.File(fn)
        self.assertEqual(f['block0000']['data'].dtype, np.float32)
        f.close()
        _, data, _ = gtsdf.load(fn)
        np.testing.assert_array_equal(data, np.arange(12).reshape(6, 2))

    def test_all(self):
        fn = tmp_path + "all.hdf5"
        gtsdf.save(fn, np.arange(12).reshape(6, 2),
                   name='datasetname',
                   time=range(6), time_step=1 / 4, time_start=4,
                   description='datasetdescription',
                   attribute_names=['att1', 'att2'],
                   attribute_units=['s', 'm/s'],
                   attribute_descriptions=['d1', 'd2'])
        time, data, info = gtsdf.load(fn)
        self.assertEqual(info['name'], "datasetname")
        self.assertEqual(info['type'], "General time series data format")
        self.assertEqual(info['description'], "datasetdescription")
        self.assertEqual(list(info['attribute_names']), ['att1', 'att2'])
        self.assertEqual(list(info['attribute_units']), ['s', 'm/s'])
        self.assertEqual(list(info['attribute_descriptions']), ['d1', 'd2'])
        np.testing.assert_array_equal(time, np.arange(4, 5.5, .25))
        np.testing.assert_array_almost_equal(data, np.arange(12).reshape(6, 2), 4)

    def test_append(self):
        fn = tmp_path + 'append.hdf5'
        d = np.arange(48, dtype=np.float32).reshape(24, 2)
        d[2, 0] = np.nan
        gtsdf.save(fn, d)
        _, data, _ = gtsdf.load(fn)
        np.testing.assert_array_almost_equal(data, d, 3)
        gtsdf.append_block(fn, d)
        _, data, _ = gtsdf.load(fn)
        self.assertEqual(data.shape, (48, 2))
        np.testing.assert_array_almost_equal(data, np.append(d, d, 0), 3)
        f = h5py.File(fn)
        self.assertIn('gains', f['block0001'])
        f.close()

    def test_append_small_block(self):
        fn = tmp_path + 'append_small_block.hdf5'
        d = np.arange(12, dtype=np.float32).reshape(2, 6)
        gtsdf.save(fn, d)
        gtsdf.append_block(fn, d + 12)
        f = h5py.File(fn)
        self.assertNotIn('gains', f['block0001'])
        f.close()

    def test_nan_float(self):
        fn = tmp_path + 'nan.hdf5'
        d = np.arange(12, dtype=np.float32).reshape(6, 2)
        d[2, 0] = np.nan
        gtsdf.save(fn, d)
        _, data, _ = gtsdf.load(fn)
        np.testing.assert_array_almost_equal(data, d, 4)

    def test_outlier(self):
        fn = tmp_path + 'outlier.hdf5'
        d = np.arange(12, dtype=np.float32).reshape(6, 2)
        d[2, 0] = 10 ** 4
        d[3, 1] = 10 ** 4
        self.assertRaises(Warning, gtsdf.save, fn, d)
        _, data, _ = gtsdf.load(fn)

    def test_inf(self):
        fn = tmp_path + 'outlier.hdf5'
        d = np.arange(12, dtype=np.float32).reshape(6, 2)
        d[2, 0] = np.inf
        d[3, 1] = 10 ** 3
        self.assertRaises(ValueError, gtsdf.save, fn, d)

    def test_loadpandas(self):
        fn = tmp_path + "all.hdf5"
        gtsdf.save(fn, np.arange(12).reshape(6, 2),
                   name='datasetname',
                   time=range(6), time_step=1 / 4, time_start=4,
                   description='datasetdescription',
                   attribute_names=['att1', 'att2'],
                   attribute_units=['s', 'm/s'],
                   attribute_descriptions=['d1', 'd2'])
        df = gtsdf.load_pandas(fn)

    def test_loadtesthdf5(self):
        time, data, info = gtsdf.load(tfp + 'test.hdf5')
        self.assertEqual(time[1], 0.05)
        self.assertEqual(data[1, 1], 11.986652374267578)
        self.assertEqual(info['attribute_names'][1], "WSP gl. coo.,Vy")

    def test_loadhdf5File(self):
        f = h5py.File(tfp + 'test.hdf5')
        time, data, info = gtsdf.load(f)

        self.assertEqual(time[1], 0.05)
        self.assertEqual(data[1, 1], 11.986652374267578)
        self.assertEqual(info['attribute_names'][1], "WSP gl. coo.,Vy")

    def test_gtsdf_dataset(self):
        ds = gtsdf.Dataset(tfp + 'test.hdf5')
        self.assertEqual(ds.data.shape, (2440, 49))
        self.assertEqual(ds('Time')[1], 0.05)
        self.assertEqual(ds.Time[1], 0.05)
        self.assertRaisesRegex(AttributeError, "'Dataset' object has no attribute 'Time1'", lambda: ds.Time1)
        self.assertEqual(ds(2)[1], 12.04148006439209)
        n = ds.info['attribute_names'][2]
        self.assertEqual(n, "WSP gl. coo.,Vy")
        self.assertEqual(ds(n)[1], 12.04148006439209)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
