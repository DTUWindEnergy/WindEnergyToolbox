'''
Created on 20. jul. 2017

@author: mmpe
'''
import unittest
from wetb.utils.test_files import get_test_file
from wetb.wind.turbulence import mann_turbulence
from wetb.wind.turbulence.mann_turbulence import parameters2name
from tests import npt


class TestMannTurbulence(unittest.TestCase):

    def testLoad(self):
        fn = get_test_file('h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat')
        self.assertRaises(AssertionError, mann_turbulence.load, fn, (31, 31))
        self.assertRaises(AssertionError, mann_turbulence.load, fn, (8192, 32, 32))
        u = mann_turbulence.load(fn, (8192, 8, 8))
        self.assertEqual(u.shape, (8192, 8 * 8))

    def test_loaduvw(self):
        fn = 'h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat'
        fn_lst = [get_test_file(fn % uvw) for uvw in 'uvw']

        u, v, w = mann_turbulence.load_uvw(fn_lst, (8192, 8, 8))
        self.assertEqual(u.shape, (8192, 8 * 8))
        self.assertTrue(u.shape == v.shape == w.shape == (8192, 8 * 8))
        u, v, w = mann_turbulence.load_uvw(fn_lst[0].replace("u.dat", "%s.dat"), (8192, 8, 8))
        self.assertEqual(u.shape, (8192, 8 * 8))
        self.assertTrue(u.shape == v.shape == w.shape == (8192, 8 * 8))

    def test_parameters2name(self):
        self.assertEqual(parameters2name((8192, 8, 8), (16384, 32, 32), 0.15, 10, 3.3, 1, 1)[
                         0], './turb/mann_l10.0_ae0.1500_g3.3_h1_8192x8x8_2.000x4.00x4.00_s0001u.turb')

    def test_save(self):
        fn = get_test_file('h2a8192_8_8_16384_32_32_0.15_10_3.3u.dat')
        u_ref = mann_turbulence.load(fn, (8192, 8, 8))
        mann_turbulence.save(u_ref, 'tmp_u.turb')
        u = mann_turbulence.load('tmp_u.turb', (8192, 8, 8))
        npt.assert_array_equal(u, u_ref)


#     def test_name2parameters(self):
#         self.assertEqual(name2parameters('./turb/mann_l10.0_ae0.15_g3.3_h1_8192x8x8_2.000x4.00x4.00_s0001u.turb'),
#                          ((8192, 8, 8), (16384., 32., 32.), 0.15, 10, 3.3, 1, 1))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
