# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:06:05 2017

@author: shfe

Note: The module contains an assortment of tools for working with linear wave theory calculations.
"""
# py2/3 compatible:
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

g = 9.806  # gravitational acceleration in m^2/s

def wave_number(g, omega, h):
    """
    Computes the wave number for given frequency and water depth
    (linear dispersion relationship)
    :param omega: -- wave frequency
    :param g: -- gravitational acceleration (defaults to 9.806 m/s^2)
    :param h: -- the water depth
    :returns k: the wave number
    """

    p = omega**2 * h / g
    q = dispersion(p)
    k = q * omega**2 / g

    return k


def frequency(g, k, h):
    """
    Computes the frequency for a given wave number and water depth
    (linear dispersion relationship)
    :param k: the wave number
    :param g: -- gravitational acceleration
    :param h: -- the water depth
    :returns omega: -- wave frequency
    """

    return np.sqrt(g * k * np.tanh(k * h))


def dispersion(p, tol=1e-14, max_iter=100):
    """
    The linear dispersion relation in non-dimensional form:
    finds q, given p
    q = gk/omega^2     non-d wave number
    p = omega^2 h / g   non-d water depth
    Starts with the Fenton and McKee approximation, then iterates with Newton's
    method until accurate to within tol.
    :param p: non-dimensional water depth
    :param tol=1e-14: acceptable tolerance
    :param max_iter: maximum number of iterations to accept
                     (it SHOULD converge in well less than 100)
    """
    if p <= 0.0:
        raise ValueError("Non dimensional water depth d must be >= 0.0")
    # First guess (from Fenton and McKee):
    q = np.tanh(p ** 0.75) ** (-2.0 / 3.0)

    iter = 0
    f = q * np.tanh(q * p) - 1
    while abs(f) > tol:
        qp = q * p
        fp = qp / (np.cosh(qp) ** 2) + np.tanh(qp)
        q = q - f / fp
        f = q * np.tanh(q * p) - 1
        iter += 1
        if iter > max_iter:
            raise RuntimeError("Maximum number of iterations reached in dispersion()")
    return q


def max_u(a, omega, g, h, z=None):
    """
    Compute the maximum Eulerian horizontal velocity at a given depth
    :param a: -- wave amplitude (1/2 the height)
    :param omega: -- wave frequency
    :param g: -- gravitational acceleration (defaults to 9.806 m/s^2)
    :param h: -- the water depth
    :param z=None: -- the depth at which to compute the maximum velocity.
                   if None, then h is used (the bottom)
    """

    z = h if z is None else z
    k = wave_number(g, omega, h)
    u = a * omega * (np.cosh(k * (h + z)) / np.sinh(k * h))

    return u


def amp_scale_at_depth(g, omega, h, z):
    """
    Compute the scale factor of the orbital amplitude at the given depth
    :param g: -- gravitational acceleration
    :param omega: -- the wave frequency
    :param h: -- the water depth
    :param z: -- the depth at which to compute the scale factor
    """

    k = wave_number(g, omega, h)

    return np.cosh(k * (h + z)) / np.cosh(k * (h))


def celerity(k, h, g=g):
    """
    Compute the celerity (wave speed, phase speed) for a given wave number and depth
    :param k: -- the wave number
    :param h: -- the water depth
    :param g=g: -- gravitational acceleration (defaults to 9.806 m/s^2)
    """

    C = np.sqrt(g / k * np.tanh(k * h))

    return C


def group_speed(k, h, g=g):
    """
    Compute the group speed for a given wave number and depth
    :param k: -- the wave number
    :param h: -- the water depth
    :param g=g: -- gravitational acceleration (defaults to 9.806 m/s^2)
    """

    n = 1.0 / 2 * (1 + (2 * k * h / np.sinh(2 * k * h)))
    Cg = n * celerity(k, h, g)

    return Cg


def shoaling_coeff(omega, g, h0, h2):
    """
    Compute the shoaling coeff for two depths: ho and h2.
    The shoaling coeff is the ratio of wave height (H2) at a particular
    point of interest to the original or deep water wave height (H0).
    Pass in h0 = None for deep water
    :param omega: -- the wave frequency
    :param g: -- gravitational acceleration
    :param h0: -- the initial water depth
    :param h2: -- the depth at which to compute the shoaling coeff
    """

    k2 = wave_number(g, omega, h2)
    Cg2 = group_speed(k2, h2, g)
    if h0 is not None:
        k0 = wave_number(g, omega, h0)
        Cg0 = group_speed(k0, h0, g)
        Ks = np.sqrt(Cg0 / Cg2)
        return Ks
    else:  # Deep water
        return np.sqrt((g / (2 * omega)) / Cg2)
