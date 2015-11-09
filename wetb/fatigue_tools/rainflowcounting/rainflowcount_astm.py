'''
Created on 27/02/2013

@author: mmpe



How to use:

import_cython("cy_rainflowcount",'cy_rainflowcount.py','')
from cy_rainflowcount import find_extremes,rainflow

ext = find_extremes(np.array([-2,0,1,0,-3,0,5,0,-1,0,3,0,-4,0,4,0,-2]).astype(np.double))
print rainflow(ext)
'''

import numpy as np


def find_extremes(signal):  #cpdef find_extremes(np.ndarray[double,ndim=1] signal):
    """return indexes of local minima and maxima plus first and last element of signal"""

    #cdef int pi, i
    # sign of gradient
    sign_grad = np.int8(np.sign(np.diff(signal)))

    # remove plateaus(sign_grad==0) by sign_grad[plateau_index]=sign_grad[plateau_index-1]
    plateau_indexes, = np.where(sign_grad == 0)
    if len(plateau_indexes) > 0 and plateau_indexes[0] == 0:
        # first element is a plateau
        if len(plateau_indexes) == len(sign_grad):
                # All values are equal to crossing level!
                return np.array([0])

        # set first element = first element which is not a plateau and delete plateau index
        i = 0
        while sign_grad[i] == 0:
            i += 1
        sign_grad[0] = sign_grad[i]

        plateau_indexes = np.delete(plateau_indexes, 0)

    for pi in plateau_indexes.tolist():
        sign_grad[pi] = sign_grad[pi - 1]

    extremes, = np.where(np.r_[1, (sign_grad[1:] * sign_grad[:-1] < 0), 1])

    return signal[extremes]


def rainflowcount(sig):  #cpdef rainflowcount(np.ndarray[double,ndim=1] sig):
    """Cython compilable rain ampl_mean count without time analysis


    This implemementation is based on the c-implementation by Adam Nieslony found at
    the MATLAB Central File Exchange http://www.mathworks.com/matlabcentral/fileexchange/3026

    References
    ----------
    Adam Nieslony, "Determination of fragments of multiaxial service loading
    strongly influencing the fatigue of machine components,"
    Mechanical Systems and Signal Processing 23, no. 8 (2009): 2712-2721.

    and is based on the following standard:
    ASTM E 1049-85 (Reapproved 1997), Standard practices for cycle counting in
    fatigue analysis, in: Annual Book of ASTM Standards, vol. 03.01, ASTM,
    Philadelphia, 1999, pp. 710-718.

    Copyright (c) 1999-2002 by Adam Nieslony

    Ported to Cython compilable Python by Mads M Pedersen
    In addition peak amplitude is changed to peak to peak amplitude


    """

    #cdef int sig_ptr, index
    #cdef double ampl
    a = []
    sig_ptr = 0
    ampl_mean = []
    for _ in range(len(sig)):
        a.append(sig[sig_ptr])
        sig_ptr += 1
        while len(a) > 2 and abs(a[-3] - a[-2]) <= abs(a[-2] - a[-1]):
            ampl = abs(a[-3] - a[-2])
            mean = (a[-3] + a[-2]) / 2;
            if len(a) == 3:
                del a[0]
                if ampl > 0:
                    ampl_mean.append((ampl, mean))
            elif len(a) > 3:
                del a[-3:-1]
                if ampl > 0:
                    ampl_mean.append((ampl, mean))
                    ampl_mean.append((ampl, mean))
    for index in range(len(a) - 1):
        ampl = abs(a[index] - a[index + 1])
        mean = (a[index] + a[index + 1]) / 2;
        if ampl > 0:
            ampl_mean.append((ampl, mean))
    return ampl_mean
