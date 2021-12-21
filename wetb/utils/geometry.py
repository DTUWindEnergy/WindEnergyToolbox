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
    axis : int
        if dir is 2d array_like, axis defines which axis to take the mean of

    Returns
    -------
    mean_deg : float
        Mean angle
    """
    return deg(mean_rad(rad(dir), axis))

def mean_rad(dir, axis=0):
    """Mean of angles in radians

    Parameters
    ----------
    dir : array_like
        Angles in radians
    axis : int
        if dir is 2d array_like, axis defines which axis to take the mean of

    Returns
    -------
    mean_rad : float
        Mean angle
    """
    return np.arctan2(np.nanmean(np.sin(dir[:]), axis), np.nanmean(np.cos(dir[:]), axis))


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
    return deg(std_rad(rad(dir)))

def std_rad(dir):
    """Standard deviation of angles in radians

    Parameters
    ----------
    dir : array_like
        Angles in radians

    Returns
    -------
    std_rad : float
        standard deviation
    """
    return np.sqrt(1 - (np.nanmean(np.sin(dir)) ** 2 + np.nanmean(np.cos(dir)) ** 2))

def rpm2rads(rpm):
    return rpm * 2 * np.pi / 60


def rads2rpm(rads):
    return rads/ (2 * np.pi) * 60