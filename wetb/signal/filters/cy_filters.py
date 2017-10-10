'''
Created on 29/05/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''


import cython
import numpy as np
#cimport numpy as np


@cython.ccall
@cython.locals(alpha=cython.float, i=cython.int)
def cy_low_pass_filter(inp, delta_t, tau):  #cpdef cy_low_pass_filter(np.ndarray[double,ndim=1] inp, double delta_t, double tau):
    #cdef np.ndarray[double,ndim=1] output
    output = np.empty_like(inp, dtype=np.float)
    output[0] = inp[0]

    alpha = delta_t / (tau + delta_t)
    for i in range(1, inp.shape[0]):
        output[i] = output[i - 1] + alpha * (inp[i] - output[i - 1])  # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]

    return output

def cy_dynamic_low_pass_filter(inp, delta_t, tau, method=1):  #cpdef cy_dynamic_low_pass_filter(np.ndarray[double,ndim=1] inp, double delta_t, np.ndarray[double,ndim=1] tau, int method=1):
    #cdef np.ndarray[double,ndim=1] output, alpha
    #cdef int i

    output = np.empty_like(inp, dtype=np.float)
    output[0] = inp[0]

    if method == 1:
        alpha = delta_t / (tau + delta_t)
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] + alpha[i] * (inp[i] - output[i - 1])  # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]
    elif method == 2:
        for i in range(1, inp.shape[0]):
            output[i] = (delta_t * (inp[i] + inp[i - 1] - output[i - 1]) + 2 * tau[i] * output[i - 1]) / (delta_t + 2 * tau[i])
    elif method == 3:
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] * np.exp(-delta_t / tau[i]) + inp[i] * (1 - np.exp(-delta_t / tau[i]))
    return output

def cy_dynamic_low_pass_filter_2d(inp, delta_t, tau, method=1):  #cpdef cy_dynamic_low_pass_filter_2d(np.ndarray[double,ndim=2] inp, double delta_t, np.ndarray[double,ndim=1] tau, int method=1):
    #cdef np.ndarray[double,ndim=2] output, alpha
    #cdef int i

    output = np.empty_like(inp, dtype=np.float)
    output[0] = inp[0]

    if method == 1:
        alpha = delta_t / (tau + delta_t)
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] + alpha[i] * (inp[i] - output[i - 1])  # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]
    elif method == 2:
        for i in range(1, inp.shape[0]):
            output[i] = (delta_t * (inp[i] + inp[i - 1] - output[i - 1]) + 2 * tau[i] * output[i - 1]) / (delta_t + 2 * tau[i])
    elif method == 3:
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] * np.exp(-delta_t / tau[i]) + inp[i] * (1 - np.exp(-delta_t / tau[i]))
    return output

def cy_dynamic_low_pass_filter_test(inp):  #cpdef cy_dynamic_low_pass_filter_test(np.ndarray[double,ndim=2] inp):
    #cdef np.ndarray[double,ndim=2] output, alpha
    #cdef int i
    output = np.empty_like(inp, dtype=np.float)
    output[0] = inp[0]
    for i in range(1, inp.shape[0]):
        output[i] = inp[i]
    return output

@cython.ccall
@cython.locals(alpha=cython.float, i=cython.int)
def cy_high_pass_filter(inp, delta_t, tau):  #cpdef cy_high_pass_filter(np.ndarray[double,ndim=1] inp, double delta_t, double tau):
    #cdef np.ndarray[double,ndim=1] output
    output = np.empty_like(inp, dtype=np.float)
    output[0] = inp[0]
    alpha = tau / (tau + delta_t)
    for i in range(1, inp.shape[0]):
        output[i] = alpha * (output[i - 1] + inp[i] - inp[i - 1])

    return output
