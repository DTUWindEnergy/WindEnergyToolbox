'''
Created on 15/01/2014

@author: MMPE
'''
import unittest

import numpy as np
from wetb.utils.geometry import rad
from wetb.utils.rotation import transformation_matrix, mdot, dots, rotate, rotate_x, \
    rotmat, rotate_y, rotate_z

x, y, z = 0, 1, 2
class Test(unittest.TestCase):


    def test_transformation_matrix(self):
        np.testing.assert_array_almost_equal(transformation_matrix(rad(30), x), [[[ 1., 0. , 0.],
                                                                                  [ 0., 0.8660254, 0.5],
                                                                                  [ 0., -0.5 , 0.8660254]]])
        np.testing.assert_array_almost_equal(transformation_matrix(rad(30), y), [[[ 0.8660254, 0. , -0.5],
                                                                     [ 0., 1., 0 ],
                                                                     [ 0.5, 0        , 0.8660254]]])
        np.testing.assert_array_almost_equal(transformation_matrix(rad(30), z), [[[ 0.8660254, 0.5 , 0.],
                                                                                  [ -0.5, 0.8660254, 0],
                                                                                  [ 0., 0 , 1]]])

    def test_rotation_matrix(self):
        np.testing.assert_array_almost_equal(rotmat(rad(30), x), [[[ 1., 0. , 0.],
                                                                   [ 0., 0.8660254, -0.5],
                                                                   [ 0., 0.5 , 0.8660254]]])
        np.testing.assert_array_almost_equal(rotmat(rad(30), y), [[[ 0.8660254, 0. , 0.5],
                                                                   [ 0., 1., 0 ],
                                                                   [ -0.5, 0        , 0.8660254]]])
        np.testing.assert_array_almost_equal(rotmat(rad(30), z), [[[ 0.8660254, -0.5 , 0.],
                                                                   [ 0.5, 0.8660254, 0],
                                                                   [ 0., 0 , 1]]])
    def test_rotation_matrixes(self):
        np.testing.assert_array_almost_equal(rotmat([rad(30), rad(60)], 0), [[[ 1., 0. , 0.],
                                                                     [ 0., 0.8660254, -0.5],
                                                                     [ 0., 0.5 , 0.8660254]],
                                                                             [[ 1., 0. , 0.],
                                                                     [ 0., 0.5, -0.8660254, ],
                                                                     [ 0., +0.8660254, 0.5 ]]])

    def test_mdot(self):
        x, y, z = 0, 1, 2
        m1 = transformation_matrix(np.pi, x)
        m2 = transformation_matrix([np.pi, np.pi / 2], y)

        np.testing.assert_array_almost_equal(m1, [[[1, 0, 0], [0, -1, 0], [0, 0, -1]]])
        np.testing.assert_array_almost_equal(mdot(m1, m1), [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        np.testing.assert_array_almost_equal(mdot(m1, m2), [[[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
        np.testing.assert_array_almost_equal(mdot(m2, m2), [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]])
        np.testing.assert_array_almost_equal(mdot(m2, m1), [[[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [[0, 0, 1], [0, -1, 0], [1, 0, 0]]])


    def test_dots(self):
        x, y, z = 0, 1, 2
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 4])

        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, x), v1).T, [[1, -3, 2]])
        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi , x), v1).T, [[1, -2, -3]])

        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, y), v1).T, [[3, 2, -1]])
        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi , y), v1).T, [[-1, 2, -3]])

        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, z), v1).T, [[-2, 1, 3]])
        np.testing.assert_array_almost_equal(dots(transformation_matrix(-np.pi , z), v1).T, [[-1, -2, 3]])

        v = np.array([v1, v2]).T

        np.testing.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], x), v).T, [[1, -3, 2], [1, -2, -4]])
        np.testing.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], y), v).T, [[3, 2, -1], [-1, 2, -4]])
        np.testing.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], z), v).T, [[-2, 1, 3], [-1, -2, 4]])
        np.testing.assert_array_almost_equal(dots([transformation_matrix(np.pi / 2, z), transformation_matrix(np.pi / 2, y), transformation_matrix(np.pi / 2, x)], v1).T, [[3, -2, 1]])


    def test_rotate(self):
        x, y, z = 0, 1, 2
        v = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi / 2, x), v), [1, -3, 2])
        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi, x), v), [1, -2, -3])

        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi / 2, y), v), [3, 2, -1])
        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi, y), v), [-1, 2, -3])

        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi / 2, z), v), [-2, 1, 3])
        np.testing.assert_array_almost_equal(rotate(rotmat(np.pi, z), v), [-1, -2, 3])

        v = np.array([[1, 2, 3], [1, 2, 4]])

        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, -np.pi], x), v)[0], [1, -3, 2])
        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, -np.pi], x), v)[1], [1, -2, -4])

        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], y), v)[0], [3, 2, -1])
        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], y), v)[1], [-1, 2, -4])

        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], z), v)[0], [-2, 1, 3])
        np.testing.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], z), v)[1], [-1, -2, 4])


    def test_rotate_xyz(self):
        x, y, z = 0, 1, 2
        v = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(rotate_x(v, np.pi / 2), [1, -3, 2])
        np.testing.assert_array_almost_equal(rotate_x(v, -np.pi), [1, -2, -3])

        np.testing.assert_array_almost_equal(rotate_y(v, np.pi / 2), [3, 2, -1])
        np.testing.assert_array_almost_equal(rotate_y(v, np.pi), [-1, 2, -3])

        np.testing.assert_array_almost_equal(rotate_z(v, np.pi / 2), [-2, 1, 3])
        np.testing.assert_array_almost_equal(rotate_z(v, np.pi), [-1, -2, 3])

        v = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_almost_equal(rotate_z(v, np.pi / 2), [[-2, 1, 3], [-5, 4, 6]])
        np.testing.assert_array_almost_equal(rotate_z(v, [np.pi / 2]), [[-2, 1, 3], [-5, 4, 6]])
        np.testing.assert_array_almost_equal(rotate_z(v, [-np.pi / 2, np.pi]), [[2, -1, 3], [-4, -5, 6]])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_rad']
    unittest.main()
