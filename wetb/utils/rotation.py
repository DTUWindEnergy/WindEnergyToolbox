'''
Created on 21/07/2014

@author: mmpe
'''
import numpy as np


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
    m = np.zeros(n * 9, np.float)
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
    m = np.zeros(n * 9, np.float)
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
        res = np.zeros_like(v, dtype=np.float)
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
    assert rotmats.shape[-1] == rotmats.shape[-2] == v.shape[-1] == 3, "rotmats is %d x %x and v is i x %d, but must be 3" % (rotmats.shape[-1], rotmats.shape[-2], v.shape[-1])

#    if isinstance(v, tuple):
#        v = np.array([v])
    n = (1, v.shape[0])[len(v.shape) == 2]

    if rotmats.shape[0] == 1:
        return np.dot(rotmats[0], v.T).T
    else:
        if v.shape[0] != rotmats.shape[0]:
            raise Exception("V and rotmats must have same first dimension")
        v = v.T
        v_rot = np.zeros_like(v, dtype=np.float)
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
    return _rotate(v, angle, lambda x, y, z, cos, sin : [x, cos * y - sin * z, sin * y + cos * z])

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
    return _rotate(v, angle, lambda x, y, z, cos, sin : [cos * x + sin * z, y, -sin * x + cos * z])

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
    return _rotate(v, angle, lambda x, y, z, cos, sin : [cos * x - sin * y, sin * x + cos * y, z])

def _rotate(v, angle, rotfunc):
    angle = np.atleast_1d(angle)
    cos, sin = np.cos(angle), np.sin(angle)
    v = np.array(v)
    if len(v.shape) == 1:
        assert angle.shape[0] == 1
        assert v.shape[0] == 3
        return np.array(rotfunc(v[0], v[1], v[2], cos, sin), dtype=np.float).T
    else:
        assert angle.shape[0] == 1 or angle.shape[0] == v.shape[0]
        assert v.shape[1] == 3
        return np.array(rotfunc(v[:, 0], v[:, 1], v[:, 2], cos, sin), dtype=np.float).T
