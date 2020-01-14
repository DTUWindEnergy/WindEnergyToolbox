'''
Created on 15/01/2014

@author: MMPE
'''
import unittest

import numpy as np
from wetb.utils.geometry import rad
from wetb.utils.rotation import transformation_matrix, mdot, dots, rotate, rotate_x, \
    rotmat, rotate_y, rotate_z, norm, axis2axis_angle, axis_angle2axis, axis2matrix, axis_angle2quaternion,\
    quaternion2matrix, quaternion2axis_angle, matrix2quaternion, s2matrix
from tests import npt
import pytest

x, y, z = 0, 1, 2


def test_s2matrix():
    npt.assert_array_equal(s2matrix('x,-y,-z'), np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T)
    npt.assert_array_equal(s2matrix('x,-z,y'), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T)


def test_transformation_matrix():
    npt.assert_array_almost_equal(transformation_matrix(rad(30), x), [[[1., 0., 0.],
                                                                       [0., 0.8660254, 0.5],
                                                                       [0., -0.5, 0.8660254]]])
    npt.assert_array_almost_equal(transformation_matrix(rad(30), y), [[[0.8660254, 0., -0.5],
                                                                       [0., 1., 0],
                                                                       [0.5, 0, 0.8660254]]])
    npt.assert_array_almost_equal(transformation_matrix(rad(30), z), [[[0.8660254, 0.5, 0.],
                                                                       [-0.5, 0.8660254, 0],
                                                                       [0., 0, 1]]])


def test_rotation_matrix():
    npt.assert_array_almost_equal(rotmat(rad(30), x), [[[1., 0., 0.],
                                                        [0., 0.8660254, -0.5],
                                                        [0., 0.5, 0.8660254]]])
    npt.assert_array_almost_equal(rotmat(rad(30), y), [[[0.8660254, 0., 0.5],
                                                        [0., 1., 0],
                                                        [-0.5, 0, 0.8660254]]])
    npt.assert_array_almost_equal(rotmat(rad(30), z), [[[0.8660254, -0.5, 0.],
                                                        [0.5, 0.8660254, 0],
                                                        [0., 0, 1]]])


def test_rotation_matrixes():
    npt.assert_array_almost_equal(rotmat([rad(30), rad(60)], 0), [[[1., 0., 0.],
                                                                   [0., 0.8660254, -0.5],
                                                                   [0., 0.5, 0.8660254]],
                                                                  [[1., 0., 0.],
                                                                   [0., 0.5, -0.8660254, ],
                                                                   [0., +0.8660254, 0.5]]])


def test_mdot():
    x, y, z = 0, 1, 2
    m1 = transformation_matrix(np.pi, x)
    m2 = transformation_matrix([np.pi, np.pi / 2], y)

    npt.assert_array_almost_equal(m1, [[[1, 0, 0], [0, -1, 0], [0, 0, -1]]])
    npt.assert_array_almost_equal(mdot(m1, m1), [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    npt.assert_array_almost_equal(
        mdot(m1, m2), [[[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
    npt.assert_array_almost_equal(mdot(m2, m2), [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [
                                  [-1, 0, 0], [0, 1, 0], [0, 0, -1]]])
    npt.assert_array_almost_equal(
        mdot(m2, m1), [[[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [[0, 0, 1], [0, -1, 0], [1, 0, 0]]])


def test_dots():
    x, y, z = 0, 1, 2
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 4])

    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, x), v1).T, [[1, -3, 2]])
    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi, x), v1).T, [[1, -2, -3]])

    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, y), v1).T, [[3, 2, -1]])
    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi, y), v1).T, [[-1, 2, -3]])

    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi / 2, z), v1).T, [[-2, 1, 3]])
    npt.assert_array_almost_equal(dots(transformation_matrix(-np.pi, z), v1).T, [[-1, -2, 3]])

    v = np.array([v1, v2]).T

    npt.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], x), v).T, [[1, -3, 2], [1, -2, -4]])
    npt.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], y), v).T, [[3, 2, -1], [-1, 2, -4]])
    npt.assert_array_almost_equal(dots(transformation_matrix([-np.pi / 2, np.pi], z), v).T, [[-2, 1, 3], [-1, -2, 4]])
    npt.assert_array_almost_equal(dots([transformation_matrix(
        np.pi / 2, z), transformation_matrix(np.pi / 2, y), transformation_matrix(np.pi / 2, x)], v1).T, [[3, -2, 1]])


def test_rotate():
    x, y, z = 0, 1, 2
    v = np.array([1, 2, 3])

    npt.assert_array_almost_equal(rotate(rotmat(np.pi / 2, x), v), [1, -3, 2])
    npt.assert_array_almost_equal(rotate(rotmat(np.pi, x), v), [1, -2, -3])

    npt.assert_array_almost_equal(rotate(rotmat(np.pi / 2, y), v), [3, 2, -1])
    npt.assert_array_almost_equal(rotate(rotmat(np.pi, y), v), [-1, 2, -3])

    npt.assert_array_almost_equal(rotate(rotmat(np.pi / 2, z), v), [-2, 1, 3])
    npt.assert_array_almost_equal(rotate(rotmat(np.pi, z), v), [-1, -2, 3])

    v = np.array([[1, 2, 3], [1, 2, 4]])

    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, -np.pi], x), v)[0], [1, -3, 2])
    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, -np.pi], x), v)[1], [1, -2, -4])

    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], y), v)[0], [3, 2, -1])
    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], y), v)[1], [-1, 2, -4])

    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], z), v)[0], [-2, 1, 3])
    npt.assert_array_almost_equal(rotate(rotmat([np.pi / 2, np.pi], z), v)[1], [-1, -2, 4])


def test_rotate_xyz():
    x, y, z = 0, 1, 2
    v = np.array([1, 2, 3])

    npt.assert_array_almost_equal(rotate_x(v, np.pi / 2), [1, -3, 2])
    npt.assert_array_almost_equal(rotate_x(v, -np.pi), [1, -2, -3])

    npt.assert_array_almost_equal(rotate_y(v, np.pi / 2), [3, 2, -1])
    npt.assert_array_almost_equal(rotate_y(v, np.pi), [-1, 2, -3])

    npt.assert_array_almost_equal(rotate_z(v, np.pi / 2), [-2, 1, 3])
    npt.assert_array_almost_equal(rotate_z(v, np.pi), [-1, -2, 3])

    v = np.array([[1, 2, 3], [4, 5, 6]])
    npt.assert_array_almost_equal(rotate_z(v, np.pi / 2), [[-2, 1, 3], [-5, 4, 6]])
    npt.assert_array_almost_equal(rotate_z(v, [np.pi / 2]), [[-2, 1, 3], [-5, 4, 6]])
    npt.assert_array_almost_equal(rotate_z(v, [-np.pi / 2, np.pi]), [[2, -1, 3], [-4, -5, 6]])


def test_norm():
    npt.assert_equal(norm([3, 4]), 5)


def test_axis2axis_angle():
    axis_angle = np.array([2.504687681842697E-002, 0.654193239950699, 0.755912599950848, np.rad2deg(3.09150937138433)])
    axis = np.array([4.43656429408, 115.877535983, 133.895130906])
    npt.assert_array_almost_equal(axis2axis_angle(axis), axis_angle)
    npt.assert_array_almost_equal(axis_angle2axis(axis2axis_angle(axis)), axis)


axis_quaternion_matrix_lst = [([30, 0, 0], [0.9659258262890683, 0.25881904510252074, 0.0, 0.0], [[1., 0., 0.],
                                                                                                 [0., 0.8660254, -0.5],
                                                                                                 [0., 0.5, 0.8660254]]),
                              ([0, 30, 0], [0.9659258262890683, 0.0, 0.25881904510252074, 0.0], [[0.8660254, 0., 0.5],
                                                                                                 [0., 1, 0],
                                                                                                 [-0.5, 0, 0.8660254]]),
                              ([0, 0, 30], [0.9659258262890683, 0, 0, 0.25881904510252074], [[0.8660254, -0.5, 0.],
                                                                                             [0.5, 0.8660254, 0],
                                                                                             [0., 0, 1]])
                              ]


@pytest.mark.parametrize("axis,quaternion,matrix", axis_quaternion_matrix_lst)
def test_axis2matrix(axis, quaternion, matrix):
    npt.assert_array_almost_equal(axis2matrix(axis, deg=True), matrix)


def test_axis_angle2axis():
    axis_angle = np.array([2.504687681842697E-002, 0.654193239950699, 0.755912599950848, np.rad2deg(3.09150937138433)])
    axis = np.array([4.43656429408, 115.877535983, 133.895130906])
    npt.assert_array_almost_equal(axis_angle2axis(axis_angle), axis)
    npt.assert_array_almost_equal(axis2axis_angle(axis_angle2axis(axis_angle)), axis_angle)


def test_axis_angle2quaternion():
    quaternion = np.array([2.518545283038265e-002, 2.518545283038264E-002, 0.657812672860355, 0.760094812138323])
    axis_angle = np.array([2.504687681842697E-002, 0.654193239950699, 0.755912599950848, 3.09150937138433])
    npt.assert_array_almost_equal(axis_angle2quaternion(axis_angle), quaternion / norm(quaternion))


def test_quaternion2axis_angle():
    axis_angle = np.array([2.504687681842697E-002, 0.654193239950699, 0.755912599950848, 3.09150937138433])
    quaternion = np.array([2.518545283038265e-002, 2.518545283038264E-002, 0.657812672860355, 0.760094812138323])
    npt.assert_array_almost_equal(quaternion2axis_angle(quaternion), axis_angle)
    npt.assert_array_almost_equal(axis_angle2quaternion(
        quaternion2axis_angle(quaternion)), quaternion / norm(quaternion))


@pytest.mark.parametrize("axis,quaternion,matrix", axis_quaternion_matrix_lst)
def test_quaternion2matrix(axis, quaternion, matrix):
    npt.assert_array_almost_equal(quaternion2matrix(quaternion), matrix)


@pytest.mark.parametrize("axis,quaternion,matrix", axis_quaternion_matrix_lst)
def test_matrix2quaternion(axis, quaternion, matrix):
    npt.assert_array_almost_equal(matrix2quaternion(matrix), quaternion)
