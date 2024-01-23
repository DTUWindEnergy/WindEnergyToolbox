'''
Created on 10/01/2015

@author: mmpe
'''
import numpy as np
from numba.core.decorators import njit


def low_pass(input, delta_t, tau, method=1):

    if isinstance(tau, (int, float)):
        return low_pass_filter(input.astype(np.float64), delta_t, tau)
    else:
        if len(input.shape) == 2:
            return dynamic_low_pass_filter_2d(input.astype(np.float64), delta_t, tau, method)

        else:
            return dynamic_low_pass_filter(input.astype(np.float64), delta_t, tau, method)


def high_pass(input, delta_t, tau):
    return high_pass_filter(input.astype(np.float64), delta_t, tau)


@njit(cache=True)
def low_pass_filter(inp, delta_t, tau):

    output = np.empty_like(inp, dtype=np.float32)
    output[0] = inp[0]

    alpha = delta_t / (tau + delta_t)
    for i in range(1, inp.shape[0]):
        # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]
        output[i] = output[i - 1] + alpha * (inp[i] - output[i - 1])

    return output


@njit(cache=True)
def dynamic_low_pass_filter(inp, delta_t, tau, method=1):

    output = np.empty_like(inp, dtype=np.float32)
    output[0] = inp[0]

    if method == 1:
        alpha = delta_t / (tau + delta_t)
        for i in range(1, inp.shape[0]):
            # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]
            output[i] = output[i - 1] + alpha[i] * (inp[i] - output[i - 1])
    elif method == 2:
        for i in range(1, inp.shape[0]):
            output[i] = (delta_t * (inp[i] + inp[i - 1] - output[i - 1]) +
                         2 * tau[i] * output[i - 1]) / (delta_t + 2 * tau[i])
    elif method == 3:
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] * np.exp(-delta_t / tau[i]) + inp[i] * (1 - np.exp(-delta_t / tau[i]))
    return output


@njit(cache=True)
def dynamic_low_pass_filter_2d(inp, delta_t, tau, method=1):

    output = np.empty_like(inp, dtype=np.float32)
    output[0] = inp[0]

    if method == 1:
        alpha = delta_t / (tau + delta_t)
        for i in range(1, inp.shape[0]):
            # Same as output[i] = alpha*inp[i]+(1-alpha)*output[i-1]
            output[i] = output[i - 1] + alpha[i] * (inp[i] - output[i - 1])
    elif method == 2:
        for i in range(1, inp.shape[0]):
            output[i] = (delta_t * (inp[i] + inp[i - 1] - output[i - 1]) +
                         2 * tau[i] * output[i - 1]) / (delta_t + 2 * tau[i])
    elif method == 3:
        for i in range(1, inp.shape[0]):
            output[i] = output[i - 1] * np.exp(-delta_t / tau[i]) + inp[i] * (1 - np.exp(-delta_t / tau[i]))
    return output


@njit(cache=True)
def high_pass_filter(inp, delta_t, tau):
    output = np.empty_like(inp, dtype=np.float32)
    output[0] = inp[0]
    alpha = tau / (tau + delta_t)
    for i in range(1, inp.shape[0]):
        output[i] = alpha * (output[i - 1] + inp[i] - inp[i - 1])

    return output
