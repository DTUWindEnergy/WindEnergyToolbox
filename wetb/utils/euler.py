'''
Created on 15/01/2014

@author: MMPE
'''

import numpy as np
from wetb.utils.geometry import deg
import warnings


def Ax(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]])


def Ay(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, 0, sin],
                     [0, 1, 0],
                     [-sin, 0, cos]])


def Az(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def euler2A(euler_param):
    warnings.warn("deprecated, use wetb.rotation.quaternion2matrix instead", DeprecationWarning)
    assert len(euler_param) == 4
    e = euler_param / np.sqrt(np.sum(euler_param**2))
    e2 = e**2
    return np.array([[e2[0] + e2[1] - e2[2] - e2[3], 2 * (e[1] * e[2] + e[0] * e[3]), 2 * (e[1] * e[3] - e[0] * e[2])],
                     [2 * (e[1] * e[2] - e[0] * e[3]), e2[0] - e2[1] + e2[2] - e2[3], 2 * (e[2] * e[3] + e[0] * e[1])],
                     [2 * (e[1] * e[3] + e[0] * e[2]), 2 * (e[2] * e[3] - e[0] * e[1]), e2[0] - e2[1] - e2[2] + e2[3]]]).T


def A2euler(A):
    # method from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    warnings.warn("deprecated, use wetb.rotation.matrix2quaternion instead", DeprecationWarning)
    sqrt = np.sqrt
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = A
    tr = m00 + m11 + m22

    if (tr > 0):
        S = sqrt(tr + 1.0) * 2  # // S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif ((m00 > m11) and (m00 > m22)):
        S = sqrt(1.0 + m00 - m11 - m22) * 2  # // S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = sqrt(1.0 + m11 - m00 - m22) * 2  # // S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = sqrt(1.0 + m22 - m00 - m11) * 2  # // S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    e = np.array([qw, qx, qy, qz])
    return e / np.sqrt(np.sum(e**2))


# def A2xyz(A):
#    if abs(A[2, 0]) != 1:
#        y = -np.arcsin(A[2, 0])
#        x = np.arctan2(A[2, 1] / np.cos(y), A[2, 2] / np.cos(y))
#        z = np.arctan2(A[1, 0] / np.cos(y), A[0, 0] / np.cos(y))
#    else:
#        z = 0
#        if A[2, 0] == -1:
#            y = np.pi / 2
#            x = z + np.arctan(A[0, 1], A[0, 2])
#        else:
#            y = -np.pi / 2
#            x = -z + np.arctan(-A[0, 1], -A[0, 2])
#    return np.array([x, y, z])
#
# def zxz2euler(z1, x, z2):
#    return np.array([np.cos(.5 * (z1 + z2)) * np.cos(.5 * x),
#                     np.cos(.5 * (z1 - z2)) * np.sin(.5 * x),
#                     np.sin(.5 * (z1 - z2)) * np.sin(.5 * x),
#                     np.sin(.5 * (z1 + z2)) * np.cos(.5 * x)])
#
# def xyz2A(x, y, z):
#    return np.dot(Ax(x), np.dot(Ay(y), Az(z)))

# def euler2xyz(euler):
#    return A2xyz(euler2A(euler))

# def A2euler(A):
#    return xyz2euler(*A2xyz(A))

def euler2angle(euler):
    if euler[0] > 1:
        euler[0] = 1
    if euler[0] < -1:
        euler[0] = -1

    return np.arccos(euler[0]) * 2


def euler2gl(euler):
    return np.r_[deg(euler2angle(euler)), euler[1:]]
