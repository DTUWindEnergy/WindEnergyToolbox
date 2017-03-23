'''
Created on 08/02/2016

@author: mmpe
'''

import numpy as np




def saturated_vapor_pressure(T):
    """Calculate pressure of saturated water vapor at specified temperature as described at
    http://wahiduddin.net/calc/density_altitude.htm
    Parameters
    ---------
    t : float
        Temperature [C]

    Returns
    -------
    float
        Pressure [mb]
    """

    eso = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951 * 10 ** -2
    c2 = 0.78736169 * 10 ** -4
    c3 = -0.61117958 * 10 ** -6
    c4 = 0.43884187 * 10 ** -8
    c5 = -0.29883885 * 10 ** -10
    c6 = 0.21874425 * 10 ** -12
    c7 = -0.17892321 * 10 ** -14
    c8 = 0.11112018 * 10 ** -16
    c9 = -0.30994571 * 10 ** -19
    p = (c0 + T * (c1 + T * (c2 + T * (c3 + T * (c4 + T * (c5 + T * (c6 + T * (c7 + T * (c8 + T * (c9))))))))))
    return eso / p ** 8

def saturated_vapor_pressure2(t):
    """Calculate pressure of saturated water vapor at specified temperature as described at
    http://wahiduddin.net/calc/density_altitude.htm
    Parameters
    ---------
    t : float
        Temperature [C]

    Returns
    -------
    float
        Pressure [mb]
    """
    c0 = 6.1078
    c1 = 7.5
    c2 = 237.3
    return c0 * 10 ** ((c1 * t) / (c2 + t))

def saturated_vapor_pressure3(t):
    """Calculate pressure of saturated water vapor at specified temperature according to

    The IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use
    W. Wagner and A. Pruss
    J. Phys. Chem. Ref. Data 31, 387 (2002); http://dx.doi.org/10.1063/1.1461829

    Parameters
    ---------
    t : float
        Temperature [C]

    Returns
    -------
    float
        Pressure [mb]
    """

    T = t + 273.15
    Tc = 647.096
    Pc = 220640
    v = 1 - T / Tc
    C1 = -7.85951783
    C2 = 1.84408259
    C3 = -11.7866497
    C4 = 22.6807411
    C5 = -15.9618719
    C6 = 1.80122502
    return Pc * np.exp(Tc / T * (C1 * v + C2 * v ** 1.5 + C3 * v ** 3 + C4 * v ** 3.5 + C5 * v ** 4 + C6 * v ** 7.5))

def saturated_vapor_pressure4(t):

    """Calculate the saturated vapor pressure as described in
    http://www.vaisala.com/Vaisala%20Documents/Application%20notes/Humidity_Conversion_Formulas_B210973EN-F.pdf

    Parameters
    ---------
    t : float
        Temperature [C]

    Returns
    -------
    Saturated vapor pressure [mBar]
    """
    A = 6.116441
    m = 7.591386
    Tn = 240.7263
    return A * 10 ** ((m * t) / (t + Tn))

def saturated_vapor_pressure_IEC(t):
    """Calculate the saturated vapor pressure according to IEC 61400-12-1

    Parameters
    ---------
    t : float
        Temperature [C]

    Returns
    -------
    Saturated vapor pressure [mBar]
    """

    T = t + 273.15
    return 0.000000205 * np.exp(0.0631846 * T)



def drew_point(t, RH):
    A = 6.116441
    m = 7.591386
    Tn = 240.7263
    Pw = saturated_vapor_pressure4(t) * RH / 100
    return Tn / (m / np.log10(Pw / A) - 1)

def air_density(P, t, rh=0, saturated_vapor_pressure_function=saturated_vapor_pressure):
    """Calculate the density of atmospheric air at specified pressure, temperature and humidity
    source: http://wahiduddin.net/calc/density_altitude.htm
    Equivalent to formulation in IEC61400-12-1 if used with the saturated_vapor_pressure_IEC function


    Parameters
    ---------
    P : float
        Atmospheric pressure [mb]=[hPa]
    t : float
        Temperature [C]
    rh : float
        Relative humidity [%]
    saturated_vapor_pressure_function : function
        Function, f(t)->P, that takes the temperature in celcius as input and
        returns the saturated vapor pressure in mbar

    Returns
    -------
    float
        Density of atmospheric air
    """
    Pv = saturated_vapor_pressure_function(t) * rh / 100
    Pd = P - Pv
    Rv = 461.4964
    Rd = 287.0531
    Tk = t + 273.15
    return (Pd * 100 / (Rd * Tk)) + (Pv * 100 / (Rv * Tk))

def R(rh=0, t=15, P=1014):
    """Specific gas constant ~287.058 J/(kg K) for dry air
    
    Parameters
    ---------
    rh : float
        Relative humidity [%]
    t : float
        Temperature [C]
    P : float
        pressure [hPa]
    
    Returns
    -------
    Specific gas constant
    """
    #assert np.all((900<P)&(P<1100)), "Pressure outside range 900 to 1100"
    #assert np.all((-50<t)&(t<100)), "Temperature outside range -50 to 100"
    Tk = t + 273.15
    return P * 100 / (air_density(P, t, rh) * Tk)

if __name__ == "__main__":
    pass
    
#     #print (air_density(1013, 65, 50))
#     #print (drew_point(40, 50))
#     import matplotlib.pyplot as plt
#     if 0:
#         t = np.arange(-20, 50)
#         plt.plot(t, saturated_vapor_pressure(t))
#         plt.plot(t, saturated_vapor_pressure2(t), "--")
#         plt.plot(t, saturated_vapor_pressure3(t), ":")
#         plt.show()
# 
#     if 0:
#         t = np.arange(5, 25)
#         plt.xlabel("Temperature [C]")
#         plt.ylabel("Density [kg/m3]")
#         plt.plot(t, air_density(990, t, 50), label='P: 990 hPa')
#         plt.plot(t, air_density(1018, t, 50), label='P: 1018 hPa')
#         plt.legend()
#         plt.show()
# 
#     p = 1013
#     t = 20
#     rh = 70
#     print (R(p, t, rh) * (t + 273.15))
#     print ((p * 100) / air_density(p, t, rh))
#     
#     print (R(p,10,80))
#     print (R(p,100,80))
# 
#     if 0:
#         rh = np.arange(0, 100)
#         plt.xlabel("Rh [%]")
#         plt.ylabel("R")
#         plt.plot(rh, R(990, 20, rh), label='P: 990 hPa, 20 t')
# 
#         plt.legend()
#         plt.show()
