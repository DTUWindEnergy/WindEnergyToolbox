#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:11:48 2025

@author: dave
"""
import unittest
import numpy as np
import pandas as pd
from wetb.hawc2.sensor_names import unified_channel_names
from wetb.hawc2.Hawc2io import ReadHawc2
import os


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/hawc2io/')  # test file path

class TestHAWC2SensorNames(unittest.TestCase):

    def test_unified_channel_names(self):

        # only read the sel file (not the actual data)
        fname = os.path.join(testfilepath, 'hawc2bin_chantest_3.sel')
        res = ReadHawc2(fname)
        ch_dict, ch_df = unified_channel_names(res.ChInfo)

        # when loading the reference result from csv everything will be a string
        # so this makes the comparison easier. What is missing in unified_channel_names
        # is also casting each column to the correct data types.
        ch_df = ch_df.astype(str)
        ch_df_ref = pd.read_csv(fname.replace('.sel', '.ch_df.csv'), sep=',',
                                dtype=str, keep_default_na=False, index_col=0)

        np.testing.assert_equal(ch_df.columns.values, ch_df_ref.columns.values)
        np.testing.assert_equal(ch_df.values, ch_df_ref.values)


if __name__ == "__main__":
    unittest.main()
