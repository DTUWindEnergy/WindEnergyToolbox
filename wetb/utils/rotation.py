'''
Created on 21/07/2014

@author: mmpe
'''
import numpy as np


def s2matrix(s):
    """Creates a transformation matrix from string

    Parameters
    ----------
    s : string
        x is 
    Returns
    -------
    matrix : array_like
        3x3 transformation matrix             

    Examples
    --------
    >> s2matrix('x,y,z') # identity matrix
    >> s2matrix('x,-z,y) # 90 deg rotation around x

    """
    d = {xyz: v for xyz, v in zip('xyz', np.eye(3))}
    return np.array([eval(xyz, d) for xyz in s.split(",")]).T


def transformation_matrix(angles, xyz):
    """Create Transformation matrix(es)
    !!!Note that the columns of the returned matrix(es) is original(unrotate) xyz-axes in rotated coordinates\n
    !!!Multiplying a this matrix by a vector rotates the vector -angle radians (in right handed terminology).

    Parameters
    ----------
    angles : int, float or array_like
        Angle(s) for rotation matrix(es). \n
        One rotation matrix will be returned for each angle
    xyz : {0,1,2}
        - 0: rotation around x(first) axis\n
        - 1: rotation around y(second) axis\n
        - 2: rotation around z(third) axis\n

    Returns
    -------
    transformation_matrix : array_like, shape = (no_angles, 3,3)
        Rotation matrix(es)
    """
    indexes = [(0, 4, 8, 5, 7), (4, 0, 8, 6, 2), (8, 0, 4, 1, 3)]  # 1, cos,cos,sin,-sin for x,y and z rotation matrix
    if isinstance(angles, (int, float)):
        n = 1
    else:
        n = len(angles)
    m = np.zeros(n * 9, float)
    cosx = np.cos(angles)
    sinx = np.sin(angles)
    m[indexes[xyz][0]::9] = 1
    m[indexes[xyz][1]::9] = cosx
    m[indexes[xyz][2]::9] = cosx
    m[indexes[xyz][3]::9] = sinx
    m[indexes[xyz][4]::9] = -sinx
    return m.reshape(n, 3, 3)


def rotmat(angles, xyz):
    """Create rotation matrix(es)
    !!!Note that the columns of the returned matrix(es) is rotated xyz-axes in original(unrotated) coordinates\n
    Multiplying a this matrix by a vector rotates the vector +angle radians (in right handed terminology).
    Parameters
    ----------
    angles : int, float or array_like
        Angle(s) for rotation matrix(es). \n
        Note that
        One rotation matrix will be returned for each angle
    xyz : {0,1,2}
        - 0: rotation around x(first) axis\n
        - 1: rotation around y(second) axis\n
        - 2: rotation around z(third) axis\n

    Returns
    -------
    transformation_matrix : array_like, shape = (no_angles, 3,3)
        Rotation matrix(es)
    """
    indexes = [(0, 4, 8, 5, 7), (4, 0, 8, 6, 2), (8, 0, 4, 1, 3)]  # 1, cos,cos,sin,-sin for x,y and z rotation matrix
    if isinstance(angles, (int, float)):
        n = 1
    else:
        n = len(angles)
    m = np.zeros(n * 9, float)
    cosx = np.cos(angles)
    sinx = np.sin(angles)
    m[indexes[xyz][0]::9] = 1
    m[indexes[xyz][1]::9] = cosx
    m[indexes[xyz][2]::9] = cosx
    m[indexes[xyz][3]::9] = -sinx
    m[indexes[xyz][4]::9] = sinx
    return m.reshape(n, 3, 3)


def mdot(m1, m2):
    """Multiplication of matrix pairs

    Parameters
    ----------
    m1 : array_like shape = (i, n, m)
        First matrix set, i.e. i matrixes, size n x m\n
        i must be equal to j or 1
    m2 : array_like, shape = (j,m,p)
        Second matrix set, i.e. j matrixes, size m x p\n
        j must be equal to i or 1

    Returns
    -------
    mdot : array_like, shape(max(i,j),n,p)
        Matrix products
    """
    if m1.shape[0] == 1 and m2.shape[0] == 1:
        return np.array([np.dot(m1[0], m2[0])])
    elif m1.shape[0] > 1 and m2.shape[0] == 1:
        mprod = np.empty_like(m1)
        for i in range(m1.shape[0]):
            mprod[i, :] = np.dot(m1[i, :], m2[0])
    elif m1.shape[0] == 1 and m2.shape[0] > 1:
        mprod = np.empty_like(m2)
        for i in range(m2.shape[0]):
            mprod[i, :] = np.dot(m1[0], m2[i, :])
    elif m1.shape[0] > 1 and m2.shape[0] > 1 and m1.shape[0] == m2.shape[0]:
        mprod = np.empty_like(m1)
        for i in range(m1.shape[0]):
            mprod[i, :] = np.dot(m1[i, :], m2[i, :])
    else:
        raise Exception("m1 and m2 must have same first dimension or one of the matrices must have first dimension 1")
    return mprod


def dots(rotmats, v):
    """Rotate vector(s) v by rotation matrix(es)

    Parameters
    ----------
    rotmats : array_like, shape=(i,3,3) or list of array_like, shape=(i,3,3)
        if list of array_like, rotation matrixes are recursively reduced by multiplication\n
        i must be 1 or equal to j
    vector : array_like, shape=(j,3)
        vectors to rotate

    Returns
    -------
    dots : array_like, shape=(j,3)
    """
    if isinstance(rotmats, list):
        if len(rotmats) > 1:
            rotmats_red = [mdot(rotmats[0], rotmats[1])]
            rotmats_red.extend(rotmats[2:])
            return dots(rotmats_red, v)
        else:
            m = rotmats[0]
    else:
        m = rotmats
    if len(v.shape) == 1:
        v = np.array([v]).T

    if m.shape[0] == 1:
        return np.dot(m[0], v)
    else:
        if m.shape[0] != v.shape[1]:
            raise Exception("m must have same dimension as v has number of columns")
        res = np.zeros_like(v, dtype=float)
        for i in range(v.shape[1]):
            res[:, i] = np.dot(m[i], v[:, i])
        return res


def rotate(rotmats, v):
    global x, y, z
    v = np.array(v)
    if isinstance(rotmats, (list, tuple)):
        if len(rotmats) > 1:
            rotmats_red = [mdot(rotmats[0], rotmats[1])]
            rotmats_red.extend(rotmats[2:])
            return rotate(rotmats_red, v)
        else:
            rotmats = rotmats[0]
    assert rotmats.shape[-1] == rotmats.shape[-2] == v.shape[-1] == 3, "rotmats is %d x %x and v is i x %d, but must be 3" % (
        rotmats.shape[-1], rotmats.shape[-2], v.shape[-1])

#    if isinstance(v, tuple):
#        v = np.array([v])
    n = (1, v.shape[0])[len(v.shape) == 2]

    if rotmats.shape[0] == 1:
        return np.dot(rotmats[0], v.T).T
    else:
        if v.shape[0] != rotmats.shape[0]:
            raise Exception("V and rotmats must have same first dimension")
        v = v.T
        v_rot = np.zeros_like(v, dtype=float)
        for i in range(n):
            v_rot[:, i] = np.dot(rotmats[i], v[:, i])
        return v_rot.T


def rotate_x(v, angle):
    """Rotate vector(s) around x axis

    Parameters
    ---------
    v : array_like, shape=(3,) or shape=(1-N,3)
        Vector(s) to rotate
    angle : int, float or array_like shape=(1,) or shape=(N,)
        Angle(s) [rad] to rotate

    Returns
    -------
    y : array_like
        Rotated vector(s)
    """
    return _rotate(v, angle, lambda x, y, z, cos, sin: [x, cos * y - sin * z, sin * y + cos * z])


def rotate_y(v, angle):
    """Rotate vector(s) around y axis

    Parameters
    ---------
    v : array_like, shape=(3,) or shape=(1-N,3)
        Vector(s) to rotate
    angle : int, float or array_like shape=(1,) or shape=(N,)
        Angle(s) [rad] to rotate

    Returns
    -------
    y : array_like
        Rotated vector(s)
    """
    return _rotate(v, angle, lambda x, y, z, cos, sin: [cos * x + sin * z, y, -sin * x + cos * z])


def rotate_z(v, angle):
    """Rotate vector(s) around z axis

    Parameters
    ---------
    v : array_like, shape=(3,) or shape=(1-N,3)
        Vector(s) to rotate
    angle : int, float or array_like shape=(1,) or shape=(N,)
        Angle(s) [rad] to rotate

    Returns
    -------
    y : array_like
        Rotated vector(s)
    """
    return _rotate(v, angle, lambda x, y, z, cos, sin: [cos * x - sin * y, sin * x + cos * y, z])


def _rotate(v, angle, rotfunc):
    angle = np.atleast_1d(angle)
    cos, sin = np.cos(angle), np.sin(angle)
    v = np.array(v)
    if len(v.shape) == 1:
        assert angle.shape[0] == 1
        assert v.shape[0] == 3
        return np.array(rotfunc(v[0], v[1], v[2], cos, sin), dtype=float).T
    else:
        assert angle.shape[0] == 1 or angle.shape[0] == v.shape[0]
        assert v.shape[1] == 3
        return np.array(rotfunc(v[:, 0], v[:, 1], v[:, 2], cos, sin), dtype=float).T


#=======================================================================================================================
# Conversions
#=======================================================================================================================
# http://www.euclideanspace.com/maths/geometry/rotations/conversions/
# - axis: [x,y,z]*angle_deg
# - axis_angle: [x,y,z,angle_rad]
# - Quaternion: [qw,qx,qy,qz]
# - Matrix 3x3 transformation matrix

def norm(vector):
    return np.sqrt(np.sum(np.asarray(vector)**2))


#=======================================================================================================================
# axis to ...
#=======================================================================================================================
def axis2axis_angle(axis):
    """
    deg : boolean
        if True, axis length is assumed to be angle in deg 
    """
    axis = np.asarray(axis)
    angle = np.sqrt(((axis**2).sum()))
    return np.r_[axis / np.sqrt((axis**2).sum()), angle]


def axis2matrix(axis, deg=False):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    axis = np.asarray(axis)
    angle = np.sqrt(((axis**2).sum()))
    if deg:
        angle = np.deg2rad(angle)
    if angle == 0:
        return np.eye(3)
    x, y, z = xyz = axis / np.sqrt((axis**2).sum())

    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c

    return np.array([[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                     [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                     [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])
    # alternative implementation
    #asym = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    # return c * np.eye(3, 3) + s * asym + (1 - c) * xyz[np.newaxis] * xyz[:, np.newaxis]


#=======================================================================================================================
# # axis_angle to ...
#=======================================================================================================================


def axis_angle2axis(axis_angle):
    x, y, z, angle = axis_angle
    return np.array([x, y, z]) * angle


def axis_angle2quaternion(axis_angle, deg=False):
    x, y, z, angle = axis_angle
    if deg:
        angle = np.deg2rad(angle)
    s = np.sin(angle / 2)
    x = x * s
    y = y * s
    z = z * s
    w = np.cos(angle / 2)
    return w, x, y, z


#=======================================================================================================================
# # quaternion to ...
#=======================================================================================================================

def quaternion2axis_angle(quaternion, deg=False):
    qw, qx, qy, qz = quaternion / norm(quaternion)
    angle = 2 * np.arccos(qw)
    if deg:
        angle = np.rad2deg(angle)
    t = np.sqrt(1 - qw**2)
    x = qx / t
    y = qy / t
    z = qz / t
    return x, y, z, angle


def quaternion2matrix(quaternion):
    q = quaternion / norm(quaternion)
    qw, qx, qy, qz = q
    sqw, sqx, sqy, sqz = q**2

    qxw = qx * qw
    qxy = qx * qy
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qzw = qz * qw

    return np.array([[sqx - sqy - sqz + sqw, 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)],
                     [2.0 * (qxy + qzw), -sqx + sqy - sqz + sqw, 2.0 * (qyz - qxw)],
                     [2.0 * (qxz - qyw), 2.0 * (qyz + qxw), -sqx - sqy + sqz + sqw]])

#=======================================================================================================================
# Matrix to ...
#=======================================================================================================================


def matrix2quaternion(matrix):
    # method from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    sqrt = np.sqrt
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = matrix
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


def matrix2axis_angle(matrix, deg=False):
    return quaternion2axis_angle(matrix2quaternion(matrix), deg)


def matrix2axis(matrix, deg=False):
    return axis_angle2axis(matrix2axis_angle(matrix, deg))
