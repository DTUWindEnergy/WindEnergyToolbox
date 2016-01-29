from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import map
from future import standard_library
standard_library.install_aliases()
import numpy as np

def rad(deg):
    return deg * np.pi / 180

def deg(rad):
    return rad / np.pi * 180

def sind(dir_deg):
    return np.sin(rad(dir_deg))

def cosd(dir_deg):
    return np.cos(rad(dir_deg))

def tand(dir_deg):
    return np.tan(rad(dir_deg))

def mean_deg(dir, axis=0):
    """Mean of angles in degrees

    Parameters
    ----------
    dir : array_like
        Angles in degrees

    Returns
    -------
    mean_deg : float
        Mean angle
    """
    return deg(np.arctan2(np.mean(sind(dir[:]), axis), np.mean(cosd(dir[:]), axis)))

def std_deg(dir):
    """Standard deviation of angles in degrees

    Parameters
    ----------
    dir : array_like
        Angles in degrees

    Returns
    -------
    std_deg : float
        standard deviation
    """
    return deg(np.sqrt(1 - (np.mean(sind(dir)) ** 2 + np.mean(cosd(dir)) ** 2)))


def wsp_dir2uv(wsp, dir, dir_ref=None):
    """Convert horizontal wind speed and direction to u,v

    Parameters
    ----------
    wsp : array_like
        Horizontal wind speed
    dir : array_like
        Wind direction
    dir_ref : int or float, optional
        Reference direction\n
        If None, default, the mean direction is used as reference

    Returns
    -------
    u : array_like
        u wind component
    v : array_like
        v wind component
    """
    if dir_ref is None:
        dir = dir[:] - mean_deg(dir[:])
    else:
        dir = dir[:] - dir_ref
    u = np.cos(rad(dir)) * wsp[:]
    v = -np.sin(rad(dir)) * wsp[:]
    return np.array([u, v])

def wsp_dir_tilt2uvw(wsp, dir, tilt, wsp_horizontal, dir_ref=None):
    """Convert horizontal wind speed and direction to u,v,w

    Parameters
    ----------
    wsp : array_like
        - if wsp_horizontal is True: Horizontal wind speed, $\sqrt{u^2+v^2}\n
        - if wsp_horizontal is False: Wind speed, $\sqrt{u^2+v^2+w^2}
    dir : array_like
        Wind direction
    tilt : array_like
        Wind tilt
    wsp_horizontal : bool
        See wsp
    dir_ref : int or float, optional
        Reference direction\n
        If None, default, the mean direction is used as reference


    Returns
    -------
    u : array_like
        u wind component
    v : array_like
        v wind component
    w : array_like
        v wind component
    """
    wsp, dir, tilt = wsp[:], dir[:], tilt[:]
    if wsp_horizontal:
        w = tand(tilt) * wsp
        u, v = wsp_dir2uv(wsp, dir, dir_ref)
    else:
        w = sind(tilt) * wsp
        u, v = wsp_dir2uv(np.sqrt(wsp ** 2 - w ** 2), dir, dir_ref)
    return np.array([u, v, w])



def xyz2uvw(x, y, z, left_handed=True):
    """Convert sonic x,y,z measurements to u,v,w wind components

    Parameters
    ----------
    x : array_like
        Sonic x component
    y : array_like
        Sonic x component
    z : array_like
        Sonic x component
    left_handed : boolean
        if true (default), xyz are defined in left handed coodinate system (default for some sonics)
        if false, xyz are defined in normal right handed coordinate system

    Returns
    -------
    u : array_like
        u wind component
    v : array_like
        v wind component
    w : array_like
        w wind component
    """
    x, y, z = map(np.array, [x, y, z])
    if left_handed:
        y *= -1
    theta = deg(np.arctan2(np.mean(y), np.mean(x)))
    SV = cosd(theta) * y - sind(theta) * x
    SUW = cosd(theta) * x + sind(theta) * y

    #% rotation around y of tilt
    tilt = deg(np.arctan2(np.mean(z), np.mean(SUW)))
    SU = SUW * cosd(tilt) + z * sind(tilt);
    SW = z * cosd(tilt) - SUW * sind(tilt);

    return np.array([SU, SV, SW])

def rpm2rads(rpm):
    return rpm * 2 * np.pi / 60


def abvrel2xyz(alpha, beta, vrel):
    """Convert pitot tube alpha, beta and relative velocity to local Cartesian wind speed velocities

    Parameters
    ----------
    alpha : array_like
        Pitot tube angle of attack
    beta : array_like
        Pitot tube side slip angle
    vrel : array_like
        Pitot tube relative velocity

    Returns
    -------
    x : array_like
        Wind component towards pitot tube (positive for postive vrel and -90<beta<90)
    y : array_like
        Wind component in alpha plane (positive for positive alpha)
    z : array_like
        Wind component in beta plane (positive for positive beta)
    """
    alpha = np.array(alpha, dtype=np.float)
    beta = np.array(beta, dtype=np.float)
    vrel = np.array(vrel, dtype=np.float)

    sign_vsx = -((np.abs(beta) > np.pi / 2) * 2 - 1)  # +1 for |beta| <= 90, -1 for |beta|>90
    sign_vsy = np.sign(alpha)  #+ for alpha > 0
    sign_vsz = -np.sign(beta)  #- for beta>0


    x = sign_vsx * np.sqrt(vrel ** 2 / (1 + np.tan(alpha) ** 2 + np.tan(beta) ** 2))

    m = alpha != 0
    y = np.zeros_like(alpha)
    y[m] = sign_vsy * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(alpha[m])) ** 2 + 1 + (np.tan(beta[m]) / np.tan(alpha[m])) ** 2))

    m = beta != 0
    z = np.zeros_like(alpha)
    z[m] = sign_vsz * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(beta[m])) ** 2 + 1 + (np.tan(alpha[m]) / np.tan(beta[m])) ** 2))

    return x, y, z
