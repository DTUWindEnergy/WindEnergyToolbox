'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import os

import numpy as np

from wetb.prepost.windIO import LoadResults


class TestsLoadResults(unittest.TestCase):

    def setUp(self):
        self.respath = os.path.join(os.path.dirname(__file__),
                                    '../../hawc2/tests/test_files/hawc2io/')
        self.fascii = 'Hawc2ascii'
        self.fbin = 'Hawc2bin'

    def loadresfile(self, resfile):
        res = LoadResults(self.respath, resfile)
        self.assertTrue(hasattr(res, 'sig'))
        self.assertEqual(res.Freq, 40.0)
        self.assertEqual(res.N, 800)
        self.assertEqual(res.Nch, 28)
        self.assertEqual(res.Time, 20.0)
        self.assertEqual(res.sig.shape, (800, 28))
        return res

    def test_load_ascii(self):
        res = self.loadresfile(self.fascii)
        self.assertEqual(res.FileType, 'ASCII')

    def test_load_binary(self):
        res = self.loadresfile(self.fbin)
        self.assertEqual(res.FileType, 'BINARY')

    def test_compare_ascii_bin(self):
        res_ascii = LoadResults(self.respath, self.fascii)
        res_bin = LoadResults(self.respath, self.fbin)

        for k in range(res_ascii.sig.shape[1]):
            np.testing.assert_allclose(res_ascii.sig[:,k], res_bin.sig[:,k],
                                       rtol=1e-02, atol=0.001)

    def test_unified_chan_names(self):
        res = LoadResults(self.respath, self.fascii, readdata=False)
        self.assertFalse(hasattr(res, 'sig'))

        np.testing.assert_array_equal(res.ch_df.index.values, np.arange(0,28))
        self.assertEqual(res.ch_df.ch_name.values[0], 'Time')
        self.assertEqual(res.ch_df.ch_name.values[27],
                         'windspeed-global-Vy--2.50-1.00--52.50')


if __name__ == "__main__":
    unittest.main()
