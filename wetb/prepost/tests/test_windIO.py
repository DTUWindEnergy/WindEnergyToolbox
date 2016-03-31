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
import struct

import numpy as np
import scipy.io as sio

from wetb.prepost.windIO import Turbulence, LoadResults

# path for test data files
fpath = os.path.join(os.path.dirname(__file__), 'data/')

class TestsLoadResults(unittest.TestCase):

    def setUp(self):
        pass

    def test_load(self):

        respath = '../../hawc2/tests/test_files/hawc2io/'
        resfile = 'Hawc2ascii'
        res = LoadResults(respath, resfile)



class TestsTurbulence(unittest.TestCase):

    def setUp(self):
        pass

    def print_test_info(self):
        pass

    def test_reshaped(self):
        """
        Make sure we correctly reshape the array instead of the manual
        index reassignments
        """
        fpath = 'data/turb_s100_3.00w.bin'
        fid = open(fpath, 'rb')
        turb = np.fromfile(fid, 'float32', 32*32*8192)
        turb.shape
        fid.close()
        u = np.zeros((8192,32,32))

        for i in range(8192):
            for j in range(32):
                for k in range(32):
                    u[i,j,k] = turb[ i*1024 + j*32 + k]

        u2 = np.reshape(turb, (8192, 32, 32))

        self.assertTrue(np.alltrue(np.equal(u, u2)))

    def test_headers(self):

        fpath = 'data/'

        basename = 'turb_s100_3.00_refoctave_header'
        fid = open(fpath + basename + '.wnd', 'rb')
        R1 = struct.unpack("h",fid.read(2))[0]
        R2 = struct.unpack("h",fid.read(2))[0]
        turb = struct.unpack("i",fid.read(4))[0]
        lat = struct.unpack("f",fid.read(4))[0]
        # last line
        fid.seek(100)
        LongVertComp = struct.unpack("f",fid.read(4))[0]
        fid.close()

        basename = 'turb_s100_3.00_python_header'
        fid = open(fpath + basename + '.wnd', 'rb')
        R1_p = struct.unpack("h",fid.read(2))[0]
        R2_p = struct.unpack("h",fid.read(2))[0]
        turb_p = struct.unpack("i",fid.read(4))[0]
        lat_p = struct.unpack("f",fid.read(4))[0]
        # last line
        fid.seek(100)
        LongVertComp_p = struct.unpack("f",fid.read(4))[0]
        fid.close()

        self.assertEqual(R1, R1_p)
        self.assertEqual(R2, R2_p)
        self.assertEqual(turb, turb_p)
        self.assertEqual(lat, lat_p)
        self.assertEqual(LongVertComp, LongVertComp_p)

    def test_write_bladed(self):

        fpath = 'data/'
        turb = Turbulence()
        # write with Python
        basename = 'turb_s100_3.00'
        turb.write_bladed(fpath, basename, shape=(8192,32,32))
        python = turb.read_bladed(fpath, basename)

        # load octave
        basename = 'turb_s100_3.00_refoctave'
        octave = turb.read_bladed(fpath, basename)

        # float versions of octave
        basename = 'turb_s100_3.00_refoctave_float'
        fid = open(fpath + basename + '.wnd', 'rb')
        octave32 = np.fromfile(fid, 'float32', 8192*32*32*3)

        # find the differences
        nr_diff = (python-octave).__ne__(0).sum()
        print(nr_diff)
        print(nr_diff/len(python))

        self.assertTrue(np.alltrue(python == octave))

    def test_turbdata(self):

        shape = (8192,32,32)

        fpath = 'data/'
        basename = 'turb_s100_3.00_refoctave'
        fid = open(fpath + basename + '.wnd', 'rb')

        # check the last element of the header
        fid.seek(100)
        print(struct.unpack("f",fid.read(4))[0])
        # save in a list using struct
        items = (os.path.getsize(fpath + basename + '.wnd')-104)/2
        data_list = [struct.unpack("h",fid.read(2))[0] for k in range(items)]


        fid.seek(104)
        data_16 = np.fromfile(fid, 'int16', shape[0]*shape[1]*shape[2]*3)

        fid.seek(104)
        data_8 = np.fromfile(fid, 'int8', shape[0]*shape[1]*shape[2]*3)

        self.assertTrue(np.alltrue( data_16 == data_list ))
        self.assertFalse(np.alltrue( data_8 == data_list ))

    def test_compare_octave(self):
        """
        Compare the results from the original script run via octave
        """

        turb = Turbulence()
        iu, iv, iw = turb.convert2bladed('data/', 'turb_s100_3.00',
                                         shape=(8192,32,32))
        res = sio.loadmat('data/workspace.mat')
        # increase tolerances, values have a range up to 5000-10000
        # and these values will be written to an int16 format for BLADED!
        self.assertTrue(np.allclose(res['iu'], iu, rtol=1e-03, atol=1e-2))
        self.assertTrue(np.allclose(res['iv'], iv, rtol=1e-03, atol=1e-2))
        self.assertTrue(np.allclose(res['iw'], iw, rtol=1e-03, atol=1e-2))

    def test_allindices(self):
        """
        Verify that all indices are called
        """
        fpath = 'data/turb_s100_3.00w.bin'
        fid = open(fpath, 'rb')
        turb = np.fromfile(fid, 'float32', 32*32*8192)
        turb.shape
        fid.close()

        check = []
        for i in range(8192):
            for j in range(32):
                for k in range(32):
                    check.append(i*1024 + j*32 + k)

        qq = np.array(check)
        qdiff = np.diff(qq)

        self.assertTrue(np.alltrue(np.equal(qdiff, np.ones(qdiff.shape))))


if __name__ == "__main__":
    unittest.main()
