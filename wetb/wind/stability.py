'''
Created on 05/10/2015

@author: MMPE
'''

import numpy as np
from scipy.integrate import trapz
from scipy.signal.signaltools import detrend


def MoninObukhov_length(u,v,w, T):
    """Calculate the Monin Obukhov length

    Not validated!!!

    parameters
    ----------
    u : array_like
        Horizontal wind fluctuations in mean wind direction
    v : array_like
        Horizontal wind fluctuations in perpendicular to mean wind direction
    w : array_like
        Vertical wind fluctuations
    T : array_like
        Potential temperature (close to sonic temperature)
    """
    K = 0.4
    g = 9.82
    u = detrend(u)
    u = u-np.mean(u)
    v = v-np.mean(v)
    w = w-np.mean(w)
    u_star = (np.mean(u*w)**2+np.mean(v*w)**2)**(1/4)
    wT = np.mean(w*T)
    return -u_star ** 3 * (T.mean() + 273.15) / (K * g * wT)


def L2category(L, full_category_name=False):
    """Stability category from Monin-Obukhov length
    
    Categories:
    0>L>-50: Extreme unstable (eu)
    -50>L>-100: Very unstable (vu)
    -100>L>-200: Unstable (u)
    -200>L>-500: Near unstable (nu)
    500<|L|: Neutral (n)
    200<L<500: Near stable (ns)
    50<L<200: Stable (s)
    10<L<50: Very stable (vs)
    0<L<10: Extreme stable (es)
    L=NaN: Undefined (-)
    Parameters
    ----------
    L : float or int
        Monin-Obukhov length
    full_category_name : bool, optional
        If False, default, category ids are returned, e.g. "n" for neutral
        If True, full name of category are returned, e.g. "Neutral"
    
    Returns
    -------
    Stability category : str
    
        Examples
    --------
    >>> L2category(1000)
    n 
    """
    cat_limits = np.array([-1e-99,-50,-100,-200,-500,500,200,50,10,1e-99])
    index = np.searchsorted( 1/cat_limits, 1/np.array(L))-1
    if full_category_name:
        return np.array(['Extreme unstable', 'Very unstable','Unstable','Near unstable','Neutral','Near stable','Stable','Very stable','Extreme stable','Undefined'])[index]
    else:
        return np.array(['eu', 'vu','u','nu','n','ns','s','vs','es','-'])[index]
    
def MoninObukhov_length2(u_star, w, T, specific_humidity=None):
    """Calculate the Monin Obukhov length

    Not validated!!!

    parameters
    ----------
    u_star :
        Refencence velocity at hub height
    w : array_like
        Vertical wind fluctuations
    T : array_like
        Temperature in celcius
    """
    K = 0.4
    g = 9.82
    if specific_humidity is not None:
        potential_temperature = (w * (T + 273.15)).mean() + 0.61 * (T + 273.15).mean() * (w * specific_humidity).mean()
    else:
        potential_temperature = (w * (T + 273.15)).mean()
    return -u_star ** 3 * (T.mean() + 273.15) / (K * g * potential_temperature)


def humidity_relative2specific(relative_humidity, T, P):
    """Not validated
    parameters
    ----------
    relative_humidity : float
        Relative humidity [%]
    T : float
        Temperature [C]
    P : float
        Barometric pressure [Pa]
    """
    return relative_humidity * np.exp(17.67 * T / (T + 273.15 - 29.65)) / 0.263 / P

def humidity_specific2relative2(specific_humidity, T, P):
    """Not validated
    parameters
    ----------
    specific_humidity : float
        specific_humidity [kg/kg]
    T : float
        Temperature [C]
    P : float
        Barometric pressure [Pa]
    """
    return 0.263 * P * specific_humidity / np.exp(17.67 * T / (T + 273.15 - 29.65))

if __name__ == "__main__":
    print (humidity_relative2specific(85, 8, 101325))
    print (humidity_specific2relative2(5.61, 8, 101325))
