'''
Created on 20. jul. 2017

@author: mmpe
'''
import numpy as np

from wetb.wind.turbulence.spectra import spectra, spectra_from_time_series
name_format = "mann_l%.1f_ae%.4f_g%.1f_h%d_%dx%dx%d_%.3fx%.2fx%.2f_s%04d%c.turb"


def load(filename, N=(32, 32)):
    """Load mann turbulence box

    Parameters
    ----------
    filename : str
        Filename of turbulence box
    N : tuple, (ny,nz) or (nx,ny,nz)
        Number of grid points

    Returns
    -------
    turbulence_box : nd_array

    Examples
    --------
    >>> u = load('turb_u.dat')
    """
    data = np.fromfile(filename, np.dtype('<f'), -1)
    if len(N) == 2:
        ny, nz = N
        nx = len(data) / (ny * nz)
        assert nx == int(nx), "Size of turbulence box (%d) does not match ny x nz (%d), nx=%.2f" % (
            len(data), ny * nz, nx)
        nx = int(nx)
    else:
        nx, ny, nz = N
        assert len(data) == nx * ny * \
            nz, "Size of turbulence box (%d) does not match nx x ny x nz (%d)" % (len(data), nx * ny * nz)
    return data.reshape(nx, ny * nz)


def load_uvw(filenames, N=(1024, 32, 32)):
    """Load u, v and w turbulence boxes

    Parameters
    ----------
    filenames : list or str
        if list: list of u,v,w filenames
        if str: filename pattern where u,v,w are replaced with '%s'
    N : tuple
        Number of grid point in the x, y and z direction

    Returns
    -------
    u,v,w : list of np_array

    Examples
    --------
    >>> u,v,w =load_uvw('turb_%s.dat')
    """
    if isinstance(filenames, str):
        return [load(filenames % uvw, N) for uvw in 'uvw']
    else:
        return [load(f, N) for f in filenames]


def save(turb, filename):
    np.array(turb).astype('<f').tofile(filename)


def parameters2name(no_grid_points, box_dimension, ae23, L, Gamma, high_frq_compensation, seed, folder="./turb/"):

    dxyz = tuple(np.array(box_dimension) / no_grid_points)
    return ["./turb/" + name_format % ((L, ae23, Gamma, high_frq_compensation) +
                                       no_grid_points + dxyz + (seed, uvw)) for uvw in ['u', 'v', 'w']]


def fit_mann_parameters(spatial_resolution, u, v, w=None, plt=None):
    """Fit mann parameters, ae, L, G

    Parameters
    ----------
    spatial_resolution : inf, float or array_like
        Distance between samples in meters
        - For turbulence boxes: 1/dx = Lx/Nx where dx is distance between points,
        Nx is number of points and Lx is box length in meters
        - For time series: Sample frequency / U
    u : array_like
        The u-wind component\n
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n
    v : array_like, optional
        The v-wind component
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n
    w : array_like, optional
        The w-wind component
        - if shape is (r,): One time series with *r* observations\n
        - if shape is (r,c): *c* different time series with *r* observations\n

    Returns
    -------
    ae : int or float
        Alpha epsilon^(2/3) of Mann model
    L : int or float
        Length scale of Mann model
    G : int or float
        Gamma of Mann model
    """
    from wetb.wind.turbulence.mann_parameters import fit_mann_model_spectra
    return fit_mann_model_spectra(*spectra(spatial_resolution, u, v, w), plt=plt)


def fit_mann_parameters_from_time_series(sample_frq, Uvw_lst, plt=None):
    from wetb.wind.turbulence.mann_parameters import fit_mann_model_spectra
    return fit_mann_model_spectra(*spectra_from_time_series(sample_frq, Uvw_lst), plt=plt)
