#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:58:25 2018

@author: dave
"""


import numpy as np
import scipy


def compute_env_of_env(envelope, dlc_list, Nx=300, Nsectors=12, Ntheta=181):
    """
    The function computes load envelopes for given channels and a groups of
    load cases starting from the envelopes computed for single simulations.
    The output is the envelope of the envelopes of the single simulations.
    This total envelope is projected on defined polar directions.

    Parameters
    ----------

    envelope : dict, dictionaries of interpolated envelopes of a given
                    channel (it's important that each entry of the dictonary
                    contains a matrix of the same dimensions). The dictonary
                    is organized by load case

    dlc_list : list, list of load cases

    Nx : int, default=300
        Number of points for the envelope interpolation

    Nsectors: int, default=12
        Number of sectors in which the total envelope will be divided. The
        default is every 30deg

    Ntheta; int, default=181
        Number of angles in which the envelope is interpolated in polar
        coordinates.

    Returns
    -------

    envelope : array (Nsectors x 6),
        Total envelope projected on the number of angles defined in Nsectors.
        The envelope is projected in Mx and My and the other cross-sectional
        moments and forces are fetched accordingly (at the same time step where
        the corresponding Mx and My are occuring)

    """

    # Group all the single DLCs
    cloud = np.zeros(((Nx+1)*len(envelope),6))
    for i in range(len(envelope)):
        cloud[(Nx+1)*i:(Nx+1)*(i+1),:] = envelope[dlc_list[i]]
    # Compute total Hull of all the envelopes
    hull = scipy.spatial.ConvexHull(cloud[:,:2])
    cc = np.append(cloud[hull.vertices,:2],
                   cloud[hull.vertices[0],:2].reshape(1,2),axis=0)
    # Interpolate full envelope
    cc_x,cc_up,cc_low,cc_int= int_envelope(cc[:,0], cc[:,1], Nx=Nx)
    # Project full envelope on given direction
    cc_proj = proj_envelope(cc_x, cc_up, cc_low, cc_int, Nx, Nsectors, Ntheta)

    env_proj = np.zeros([len(cc_proj),6])
    env_proj[:,:2] = cc_proj

    # Based on Mx and My, gather the remaining cross-sectional forces and
    # moments
    for ich in range(2, 6):
        s0 = np.array(cloud[hull.vertices, ich]).reshape(-1, 1)
        s1 = np.array(cloud[hull.vertices[0], ich]).reshape(-1, 1)
        s0 = np.append(s0, s1, axis=0)
        cc = np.append(cc, s0, axis=1)

        _,_,_,extra_sensor = int_envelope(cc[:,0],cc[:,ich],Nx)
        es = np.atleast_2d(np.array(extra_sensor[:,1])).T
        cc_int = np.append(cc_int,es,axis=1)

        for isec in range(Nsectors):
            ids = (np.abs(cc_int[:,0]-cc_proj[isec,0])).argmin()
            env_proj[isec,ich] = (cc_int[ids-1,ich]+cc_int[ids,ich]+\
                                                    cc_int[ids+1,ich])/3

    return env_proj


def int_envelope(ch1, ch2, Nx):
    """
    Function to interpolate envelopes and output arrays of same length

    Number of points is defined by Nx + 1, where the + 1 is needed to
    close the curve
    """

    upper = []
    lower = []

    indmax = np.argmax(ch1)
    indmin = np.argmin(ch1)
    if indmax > indmin:
        lower = np.array([ch1[indmin:indmax+1],ch2[indmin:indmax+1]]).T
        upper = np.concatenate((np.array([ch1[indmax:],ch2[indmax:]]).T,
                                np.array([ch1[:indmin+1],ch2[:indmin+1]]).T),
                                axis=0)
    else:
        upper = np.array([ch1[indmax:indmin+1],ch2[indmax:indmin+1]]).T
        lower = np.concatenate((np.array([ch1[indmin:],ch2[indmin:]]).T,
                                np.array([ch1[:indmax+1],ch2[:indmax+1]]).T),
                                axis=0)


    int_1 = np.linspace(min(upper[:,0].min(),lower[:,0].min()),
                        max(upper[:,0].max(),lower[:,0].max()),Nx/2+1)
    upper = np.flipud(upper)
    int_2_up = np.interp(int_1,np.array(upper[:,0]),np.array(upper[:,1]))
    int_2_low = np.interp(int_1,np.array(lower[:,0]),np.array(lower[:,1]))

    int_env = np.concatenate((np.array([int_1[:-1],int_2_up[:-1]]).T,
                              np.array([int_1[::-1],int_2_low[::-1]]).T),
                              axis=0)

    return int_1, int_2_up, int_2_low, int_env


def proj_envelope(env_x, env_up, env_low, env, Nx, Nsectors, Ntheta):
    """
    Function to project envelope on given angles

    Angles of projection is defined by Nsectors
    Projections are obtained in polar coordinates and outputted in
    cartesian
    """

    theta_int = np.linspace(-np.pi,np.pi,Ntheta)
    sectors = np.linspace(-np.pi,np.pi,Nsectors+1)
    proj = np.zeros([Nsectors,2])

    R_up = np.sqrt(env_x**2+env_up**2)
    theta_up = np.arctan2(env_up,env_x)

    R_low = np.sqrt(env_x**2+env_low**2)
    theta_low = np.arctan2(env_low,env_x)

    R = np.concatenate((R_up,R_low))
    theta = np.concatenate((theta_up,theta_low))
    R = R[np.argsort(theta)]
    theta = np.sort(theta)

    R_int = np.interp(theta_int,theta,R,period=2*np.pi)

    for i in range(Nsectors):
        if sectors[i]>=-np.pi and sectors[i+1]<-np.pi/2:
            indices = np.where(np.logical_and(theta_int >= sectors[i],
                                              theta_int <= sectors[i+1]))
            maxR = R_int[indices].max()
            proj[i+1,0] = maxR*np.cos(sectors[i+1])
            proj[i+1,1] = maxR*np.sin(sectors[i+1])
        elif sectors[i]==-np.pi/2:
            continue
        elif sectors[i]>-np.pi/2 and sectors[i+1]<=0:
            indices = np.where(np.logical_and(theta_int >= sectors[i],
                                              theta_int <= sectors[i+1]))
            maxR = R_int[indices].max()
            proj[i,0] = maxR*np.cos(sectors[i])
            proj[i,1] = maxR*np.sin(sectors[i])
        elif sectors[i]>=0 and sectors[i+1]<np.pi/2:
            indices = np.where(np.logical_and(theta_int >= sectors[i],
                                              theta_int <= sectors[i+1]))
            maxR = R_int[indices].max()
            proj[i+1,0] = maxR*np.cos(sectors[i+1])
            proj[i+1,1] = maxR*np.sin(sectors[i+1])
        elif sectors[i]==np.pi/2:
            continue
        elif sectors[i]>np.pi/2 and sectors[i+1]<=np.pi:
            indices = np.where(np.logical_and(theta_int >= sectors[i],
                                              theta_int <= sectors[i+1]))
            maxR = R_int[indices].max()
            proj[i,0] = maxR*np.cos(sectors[i])
            proj[i,1] = maxR*np.sin(sectors[i])

    ind = np.where(sectors==0)
    proj[ind,0] = env[:,0].max()

    ind = np.where(sectors==np.pi/2)
    proj[ind,1] = env[:,1].max()

    ind = np.where(sectors==-np.pi)
    proj[ind,0] = env[:,0].min()

    ind = np.where(sectors==-np.pi/2)
    proj[ind,1] = env[:,1].min()

    return proj


def closed_contour(cloud):
    """Return indices of the vertices of the closed convex contour that
    contain all points of the (x,y) cloud of pairs.


    Parameters
    ----------

    cloud : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to construct a convex hull from. Ndim should be
        at least 2 or higher.

    cloud_extra : np.ndarray(npoints, nr_extra_channels)


    Returns
    -------

    ivertices : ndarray(nvertices)

    vertices : ndarray(nvertices, 2+nr_extra_channels)

    """
    if not cloud.shape[1] >= 2:
        raise IndexError('Cloud dimension should be 2 or greater')

    hull = scipy.spatial.ConvexHull(cloud[:,0:2])

    # indices to the vertices containing the cloud of the first 2 dimensions
    ivertices = np.ndarray((len(hull.vertices)+1,), dtype=np.int32)
    ivertices[0:-1] = hull.vertices
    ivertices[-1] = hull.vertices[0]
    # actual vertices of all dimensions in the cloud, based on the first two
    # dimensions
    vertices = np.ndarray((len(ivertices), cloud.shape[1]))
    vertices[0:-1,:] = cloud[ivertices[0:-1],:]
    vertices[-1,:] = cloud[ivertices[-1],:]

    return vertices, ivertices


def compute_envelope(cloud, int_env=False, Nx=300):
    """
    The function computes load envelopes for given signals and a single
    load case. Starting from Mx and My moments, the other cross-sectional
    forces are identified.

    Parameters
    ----------

    x, y : np.ndarray(n)
        2 components of the same time signal, e.g x and y.

    int_env : boolean, default=False
        If the logic parameter is True, the function will interpolate the
        envelope on a given number of points

    Nx : int, default=300
        Number of points for the envelope interpolation

    Returns
    -------

    envelope : dictionary,
        The dictionary has entries refered to the channels selected.
        Inside the dictonary under each entry there is a matrix with 6
        columns, each for the sectional forces and moments

    """

    vertices, ivertices = closed_contour(cloud)

    # interpolate to a fixed location of equally spaced vertices
    if int_env:
        vert_int = np.ndarray((Nx+1, cloud.shape[1]))
        _,_,_,vert_int[:,0:2] = int_envelope(vertices[:,0], vertices[:,1], Nx)
        for i in range(2, cloud.shape[1]):
            _,_,_,extra = int_envelope(vertices[:,0], vertices[:,i], Nx)
            vert_int[:,i] = extra[:,1]
        vertices = vert_int

    return vertices

