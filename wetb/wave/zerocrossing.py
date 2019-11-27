# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:19:17 2018

@author: shfe

This script is used to plot the probability of exceedence of wave elevation for Linear and Nonlinear Wave
"""
import numpy as np
from numpy import atleast_1d, sign, mod, zeros
from wetb.wave import numba_misc
from wetb.signal.spectrum import psd
import warnings

#try:
#    from wafo import c_library as clib  # @UnresolvedImport
#except ImportError:
#    warnings.warn('c_library not found. Check its compilation.')
#    clib = None

def xor(a, b):
    """
    Return True only when inputs differ.
    """
    return a ^ b

def _findcross(x, method='numba'):
    '''Return indices to zero up and downcrossings of a vector
    '''
    
    return numba_misc.findcross(x)

def findcross(x, v=0.0, kind=None, method='clib'):
    '''
    Return indices to level v up and/or downcrossings of a vector

    Parameters
    ----------
    x : array_like
        vector with sampled values.
    v : scalar, real
        level v.
    kind : string
        defines type of wave or crossing returned. Possible options are
        'dw' : downcrossing wave
        'uw' : upcrossing wave
        'cw' : crest wave
        'tw' : trough wave
        'd'  : downcrossings only
        'u'  : upcrossings only
        None : All crossings will be returned

    Returns
    -------
    ind : array-like
        indices to the crossings in the original sequence x.

    Example
    -------
    >>> from matplotlib import pyplot as plt
    >>> import wafo.misc as wm
    >>> ones = np.ones
    >>> np.allclose(findcross([0, 1, -1, 1], 0), [0, 1, 2])
    True
    >>> v = 0.75
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,v) # all crossings
    >>> np.allclose(ind, [  9,  25,  80,  97, 151, 168, 223, 239])
    True
    >>> ind2 = wm.findcross(x,v,'u')
    >>> np.allclose(ind2, [  9,  80, 151, 223])
    True
    >>> ind3 = wm.findcross(x,v,'d')
    >>> np.allclose(ind3, [  25,  97, 168, 239])
    True
    >>> ind4 = wm.findcross(x,v,'d', method='2')
    >>> np.allclose(ind4, [  25,  97, 168, 239])
    True

    t0 = plt.plot(t,x,'.',t[ind],x[ind],'r.', t, ones(t.shape)*v)
    t0 = plt.plot(t[ind2],x[ind2],'o')
    plt.close('all')

    See also
    --------
    crossdef
    wavedef
    '''
    xn = np.int8(sign(atleast_1d(x).ravel() - v))  # @UndefinedVariable
    ind = _findcross(xn, method)
    if ind.size == 0:
        warnings.warn('No level v = %0.5g crossings found in x' % v)
        return ind

    if kind not in ('du', 'all', None):
        if kind == 'd':  # downcrossings only
            t_0 = int(xn[ind[0] + 1] > 0)
            ind = ind[t_0::2]
        elif kind == 'u':  # upcrossings  only
            t_0 = int(xn[ind[0] + 1] < 0)
            ind = ind[t_0::2]
        elif kind in ('dw', 'uw', 'tw', 'cw'):
            # make sure that the first is a level v down-crossing
            #   if kind=='dw' or kind=='tw'
            # or that the first is a level v up-crossing
            #    if kind=='uw' or kind=='cw'

            first_is_down_crossing = int(xn[ind[0]] > xn[ind[0] + 1])
            if xor(first_is_down_crossing, kind in ('dw', 'tw')):
                ind = ind[1::]

            n_c = ind.size  # number of level v crossings
            # make sure the number of troughs and crests are according to the
            # wavedef, i.e., make sure length(ind) is odd if dw or uw
            # and even if tw or cw
            is_odd = mod(n_c, 2)
            if xor(is_odd, kind in ('dw', 'uw')):
                ind = ind[:-1]
        else:
            raise ValueError('Unknown wave/crossing definition!'
                             ' ({})'.format(kind))
    return ind
    
def findtc(x_in, v=None, kind=None):
    """
    Return indices to troughs and crests of data.

    Parameters
    ----------
    x : vector
        surface elevation.
    v : real scalar
        reference level (default  v = mean of x).

    kind : string
        defines the type of wave. Possible options are
        'dw', 'uw', 'tw', 'cw' or None.
        If None indices to all troughs and crests will be returned,
        otherwise only the paired ones will be returned
        according to the wavedefinition.

    Returns
    --------
    tc_ind : vector of ints
        indices to the trough and crest turningpoints of sequence x.
    v_ind : vector of ints
        indices to the level v crossings of the original
        sequence x. (d,u)

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t)))
    >>> x1 = x[0:200,:]
    >>> itc, iv = wm.findtc(x1[:,1],0,'dw')
    >>> tc = x1[itc,:]
    >>> np.allclose(itc, [ 52, 105])
    True
    >>> itc, iv = wm.findtc(x1[:,1],0,'uw')
    >>> np.allclose(itc, [ 105, 157])
    True

    a = plt.plot(x1[:,0],x1[:,1],tc[:,0],tc[:,1],'ro')
    plt.close('all')

    See also
    --------
    findtp
    findcross,
    wavedef
    """

    x = atleast_1d(x_in)
    if v is None:
        v = x.mean()

    v_ind = findcross(x, v, kind)
    n_c = v_ind.size
    if n_c <= 2:
        warnings.warn('There are no waves!')
        return zeros(0, dtype=np.int), zeros(0, dtype=np.int)

    # determine the number of trough2crest (or crest2trough) cycles
    is_even = mod(n_c + 1, 2)
    n_tc = int((n_c - 1 - is_even) / 2)

    # allocate variables before the loop increases the speed
    ind = zeros(n_c - 1, dtype=np.int)

    first_is_down_crossing = (x[v_ind[0]] > x[v_ind[0] + 1])
    if first_is_down_crossing:
        f1, f2 = np.argmin, np.argmax
    else:
        f1, f2 = np.argmax, np.argmin

    for i in range(n_tc):
        # trough or crest
        j = 2 * i
        ind[j] = f1(x[v_ind[j] + 1:v_ind[j + 1] + 1])
        # crest or trough
        ind[j + 1] = f2(x[v_ind[j + 1] + 1:v_ind[j + 2] + 1])

    if (2 * n_tc + 1 < n_c) and (kind in (None, 'tw', 'cw')):
        # trough or crest
        ind[n_c - 2] = f1(x[v_ind[n_c - 2] + 1:v_ind[n_c - 1] + 1])

    return v_ind[:n_c - 1] + ind + 1, v_ind

def trough_crest(time, eta, v=None, wavetype=None):
    """
    Return trough and crest turning points

    Parameters
    -----------
    v : scalar
        reference level (default  v = mean of x).

    wavetype : string
        defines the type of wave. Possible options are
        'dw', 'uw', 'tw', 'cw' or None.
        If None indices to all troughs and crests will be returned,
        otherwise only the paired ones will be returned
        according to the wavedefinition.

    Returns
    --------
    tc : TurningPoints object
        with trough and crest turningpoints
    """
    ind = findtc(eta, v, wavetype)[0]
    eta_tc = eta[ind]
    
    try:
        t = time
    except Exception:
        t = ind
    t_tc = t[ind]
    
    return t_tc, eta_tc

def wave_parameters(time, eta):

    """
    Returns several wave parameters from data.
    Parameters
    ----------
    rate : scalar integer
        interpolation rate. Interpolates with spline if greater than one.
    Returns
    -------
    parameters : dict
        wave parameters such as
        Ac, At : Crest and trough amplitude, respectively
        Tc, Tt : crest time and trough time
        Hu, Hd : zero-up- and down-crossing wave height, respectively.        
        Tu, Td : zero-up- and down-crossing wave period, respectively.   
    """
    tc_ind, z_ind = findtc(eta, v=None, kind='tw')
 
    tc_a = eta[tc_ind]
    tc_t = time[tc_ind]
     
    Ac = tc_a[1::2]
    At = -tc_a[0::2]
    
    Tc = tc_t[1::2]
    Tt = tc_t[0::2]
    
    Hu = Ac + At[1:]
    Hd = Ac + At[:-1]

    tz = time[z_ind]
    tu = tz[1::2]
#    td = tz[0::2]
    Tu = np.diff(tu)
    T_crest = Tc - tu[:-1]
    return Ac, At, Tc, Tt, Hu, Hd, Tu, tu, T_crest
    