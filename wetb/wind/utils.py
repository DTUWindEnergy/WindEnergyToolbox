'''
Created on 19. dec. 2016

@author: mmpe
'''

from wetb.utils.geometry import mean_deg, rad, tand, sind, deg, cosd

import numpy as np
from scipy.signal import detrend


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
    r"""Convert horizontal wind speed and direction to u,v,w

    Parameters
    ----------
    wsp : array_like
        - if wsp_horizontal is True: Horizontal wind speed, $\sqrt{u^2+v^2}$\n
        - if wsp_horizontal is False: Wind speed, $\sqrt{u^2+v^2+w^2}$
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

#     SUW = cosd(theta) * x + sind(theta) * y
#
#     #% rotation around y of tilt
#     tilt = deg(np.arctan2(np.mean(z), np.mean(SUW)))
#     SU = SUW * cosd(tilt) + z * sind(tilt);
#     SW = z * cosd(tilt) - SUW * sind(tilt);

    SU = cosd(theta) * x + sind(theta) * y
    SW = z

    return np.array([SU, SV, SW])


def abvrel2xyz_old(alpha, beta, vrel):
    """Convert pitot tube alpha, beta and relative velocity to local Cartesian wind speed velocities

    Parameters
    ----------
    alpha : array_like
        Pitot tube angle of attack [rad]. Zero: Parallel to pitot tube. Positive: Flow from wind side (pressure side)
    beta : array_like
        Pitot tube side slip angle [rad]. Zero: Parallel to pitot tube. Positive: Flow from root side
    vrel : array_like
        Pitot tube relative velocity. Positive: flow towards pitot tube

    Returns
    -------
    x : array_like
        Wind component towards pitot tube (positive for postive vrel and -90<beta<90)
    y : array_like
        Wind component in alpha plane (positive for positive alpha)
    z : array_like
        Wind component in beta plane (positive for negative beta)
    """
    alpha = np.array(alpha, dtype=float)
    beta = np.array(beta, dtype=float)
    vrel = np.array(vrel, dtype=float)

    sign_vsx = -((np.abs(beta) > np.pi / 2) * 2 - 1)  # +1 for |beta| < 90, -1 for |beta|>90
    sign_vsy = np.sign(alpha)  # + for alpha > 0
    sign_vsz = -np.sign(beta)  # - for beta>0

    x = sign_vsx * np.sqrt(vrel ** 2 / (1 + np.tan(alpha) ** 2 + np.tan(beta) ** 2))

    m = alpha != 0
    y = np.zeros_like(alpha)
    y[m] = sign_vsy[m] * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(alpha[m])) ** 2 +
                                                 1 + (np.tan(beta[m]) / np.tan(alpha[m])) ** 2))

    m = beta != 0
    z = np.zeros_like(alpha)
    z[m] = sign_vsz[m] * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(beta[m])) ** 2 +
                                                 1 + (np.tan(alpha[m]) / np.tan(beta[m])) ** 2))

    return x, y, z


def abvrel2xyz(alpha, beta, vrel):
    """Convert pitot tube alpha, beta and relative velocity to local Cartesian wind speed velocities

    x : parallel to pitot tube, direction pitot tube root to tip, i.e. normal flow gives negative x\n
    y : component in alpha plane
    z : component in beta plane

    For typical usage where pitot tube is mounted on leading edge:\n
    x: Opposite rotational direction\n
    y: Direction of mean wind\n
    z: From blade root to tip\n


    Parameters
    ----------
    alpha : array_like
        Pitot tube angle of attack [rad]. Zero for flow towards pitot tube. Positive around z-axis. I.e.
        negative alpha (normal flow) gives positive y component
    beta : array_like
        Pitot tube side slip angle [rad]. Zero for flow towards pitot tube. Positive around y-axis. I.e.
        Positive beta (normal flow due to expansion and position in front of blade) gives positive z
    vrel : array_like
        Pitot tube relative velocity. Positive: flow towards pitot tube

    Returns
    -------
    x : array_like
        Wind component away from pitot tube (positive for postive vrel and -90<beta<90)
    y : array_like
        Wind component in alpha plane (positive for positive alpha)
    z : array_like
        Wind component in beta plane (positive for negative beta)
    """
    alpha = np.array(alpha, dtype=float)
    beta = np.array(beta, dtype=float)
    vrel = np.array(vrel, dtype=float)

    sign_vsx = ((np.abs(beta) > np.pi / 2) * 2 - 1)  # -1 for |beta| < 90, +1 for |beta|>90
    sign_vsy = -np.sign(alpha)  # - for alpha > 0
    sign_vsz = np.sign(beta)  # for beta>0

    x = sign_vsx * np.sqrt(vrel ** 2 / (1 + np.tan(alpha) ** 2 + np.tan(beta) ** 2))

    m = alpha != 0
    y = np.zeros_like(alpha)
    y[m] = sign_vsy[m] * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(alpha[m])) ** 2 +
                                                 1 + (np.tan(beta[m]) / np.tan(alpha[m])) ** 2))

    m = beta != 0
    z = np.zeros_like(alpha)
    z[m] = sign_vsz[m] * np.sqrt(vrel[m] ** 2 / ((1 / np.tan(beta[m])) ** 2 +
                                                 1 + (np.tan(alpha[m]) / np.tan(beta[m])) ** 2))

    return np.array([x, y, z]).T


def detrend_uvw(u, v=None, w=None):
    #     def _detrend(wsp):
    #         if wsp is None:
    #             return None
    #         dwsp = np.atleast_2d(wsp.copy().T).T
    #         t = np.arange(dwsp.shape[0])
    #         A = np.vstack([t, np.ones(len(t))]).T
    #         for i in range(dwsp.shape[1]):
    #             trend, offset = np.linalg.lstsq(A, dwsp[:, i])[0]
    #             dwsp[:, i] = dwsp[:, i] - t * trend + t[-1] / 2 * trend
    #         return dwsp.reshape(wsp.shape)
    def _detrend(y):
        if y is None:
            return None
        return detrend(y)
    return [_detrend(uvw) for uvw in [u, v, w]]
