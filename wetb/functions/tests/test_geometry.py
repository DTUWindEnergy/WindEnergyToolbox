'''
Created on 15/01/2014

@author: MMPE
'''
import unittest

import wetb.gtsdf
import numpy as np
from wetb.functions.geometry import rad, deg, mean_deg, sind, cosd, std_deg, xyz2uvw, \
    wsp_dir2uv, wsp_dir_tilt2uvw, tand


class Test(unittest.TestCase):


    def test_rad(self):
        self.assertEqual(rad(45), np.pi / 4)
        self.assertEqual(rad(135), np.pi * 3 / 4)


    def test_deg(self):
        self.assertEqual(45, deg(np.pi / 4))
        self.assertEqual(135, deg(np.pi * 3 / 4))

    def test_rad_deg(self):
        for i in [15, 0.5, 355, 400]:
            self.assertEqual(i, deg(rad(i)), i)

    def test_sind(self):
        self.assertAlmostEqual(sind(30), .5)

    def test_cosd(self):
        self.assertAlmostEqual(cosd(60), .5)

    def test_tand(self):
        self.assertAlmostEqual(tand(30), 0.5773, 3)


    def test_mean_deg(self):
        self.assertEqual(mean_deg(np.array([0, 90])), 45)
        self.assertAlmostEqual(mean_deg(np.array([350, 10])), 0)


    def test_mean_deg_array(self):
        a = np.array([[0, 90], [350, 10], [0, -90]])
        np.testing.assert_array_almost_equal(mean_deg(a, 1), [45, 0, -45])
        np.testing.assert_array_almost_equal(mean_deg(a.T, 0), [45, 0, -45])


    def test_std_deg(self):
        self.assertEqual(std_deg(np.array([0, 0, 0])), 0)
        self.assertAlmostEqual(std_deg(np.array([0, 90, 180, 270])), 57.296, 2)

    def test_wspdir2uv(self):
        u, v = wsp_dir2uv(np.array([1, 1, 1]), np.array([30, 0, 330]))
        np.testing.assert_array_almost_equal(u, [0.8660, 1, 0.8660], 3)
        np.testing.assert_array_almost_equal(v, [-0.5, 0, 0.5], 3)

    def test_wspdir2uv_dir_ref(self):
        u, v = wsp_dir2uv(np.array([1, 1, 1]), np.array([30, 0, 330]), 30)
        np.testing.assert_array_almost_equal(u, [1, 0.8660, .5], 3)
        np.testing.assert_array_almost_equal(v, [0, 0.5, .8660], 3)

    def test_xyz2uvw(self):
        u, v, w = xyz2uvw([1, 1, 0], [0, 1, 1], 0, left_handed=False)
        np.testing.assert_almost_equal(u, [np.sqrt(1 / 2), np.sqrt(2), np.sqrt(1 / 2)])
        np.testing.assert_almost_equal(v, [-np.sqrt(1 / 2), 0, np.sqrt(1 / 2)])


        u, v, w = xyz2uvw([1, 1, 0], [0, 1, 1], 0, left_handed=True)
        np.testing.assert_almost_equal(u, [np.sqrt(1 / 2), np.sqrt(2), np.sqrt(1 / 2)])
        np.testing.assert_almost_equal(v, [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)])

        u, v, w = xyz2uvw(np.array([-1, -1, -1]), np.array([-0.5, 0, .5]), np.array([0, 0, 0]), left_handed=False)
        np.testing.assert_array_almost_equal(u, np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(v, np.array([.5, 0, -.5]))
        np.testing.assert_array_almost_equal(w, np.array([0, 0, 0]))

        u, v, w = xyz2uvw(np.array([.5, cosd(30), 1]), np.array([sind(60), sind(30), 0]), np.array([0, 0, 0]), left_handed=False)
        np.testing.assert_array_almost_equal(u, np.array([sind(60), 1, sind(60)]))
        np.testing.assert_array_almost_equal(v, np.array([.5, 0, -.5]))
        np.testing.assert_array_almost_equal(w, np.array([0, 0, 0]))

        u, v, w = xyz2uvw(np.array([.5, cosd(30), 1]), np.array([0, 0, 0]), np.array([sind(60), sind(30), 0]), left_handed=False)
        np.testing.assert_array_almost_equal(u, np.array([sind(60), 1, sind(60)]))
        np.testing.assert_array_almost_equal(v, np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(w, np.array([.5, 0, -.5]))


    def test_wspdir2uv2(self):
        time, data, info = wetb.gtsdf.load("test_files/SonicDataset.hdf5")
        stat, x, y, z, temp, wsp, dir, tilt = data[2:3].T  #xyz is left handed
        np.testing.assert_array_almost_equal(xyz2uvw(*wsp_dir2uv(wsp, dir), z=0), xyz2uvw(x, y, 0))

    def test_wspdirtil2uvw(self):
        time, data, info = wetb.gtsdf.load("test_files/SonicDataset.hdf5")
        stat, x, y, z, temp, wsp, dir, tilt = data[3:6].T  #xyz is left handed
        wsp = np.sqrt(wsp ** 2 + z ** 2)
        np.testing.assert_array_almost_equal(xyz2uvw(*wsp_dir_tilt2uvw(wsp, dir, tilt, wsp_horizontal=False), left_handed=False), xyz2uvw(x, y, z))

    def test_wspdirtil2uvw_horizontal_wsp(self):
        time, data, info = wetb.gtsdf.load("test_files/SonicDataset.hdf5")
        stat, x, y, z, temp, wsp, dir, tilt = data[:].T  #xyz is left handed
        np.testing.assert_array_almost_equal(xyz2uvw(*wsp_dir_tilt2uvw(wsp, dir, tilt, wsp_horizontal=True), left_handed=False), xyz2uvw(x, y, z))

        np.testing.assert_array_almost_equal(wsp_dir_tilt2uvw(wsp, dir, tilt, wsp_horizontal=True, dir_ref=180), np.array([x, -y, z]), 5)
        np.testing.assert_array_almost_equal(xyz2uvw(*wsp_dir_tilt2uvw(wsp, dir, tilt, wsp_horizontal=True), left_handed=False), xyz2uvw(x, y, z))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_rad']
    unittest.main()
