'''
Created on 30. mar. 2017

@author: mmpe
'''
import numpy as np
def fix_rotor_position(rotor_position, sample_frq, rotor_speed, polynomial_sample_length=500):
    """Rotor position fitted with multiple polynomials
    
    Parameters
    ----------
    rotor_position : array_like
        Rotor position [deg] (0-360)
    sample_frq : int or float
        Sample frequency [Hz]
    rotor_speed : array_like
        Rotor speed [RPM]
    polynomial_sample_length : int, optional
        Sample lengths of polynomial used for fit
        
    Returns
    -------
    y : nd_array
        Fitted rotor position
    """
    
    from wetb.signal.subset_mean import revolution_trigger
    
    t = np.arange(len(rotor_position))
    indexes = revolution_trigger(rotor_position[:].copy(), sample_frq, rotor_speed, max_no_round_diff=4)
    
    rp = rotor_position[:].copy()
    
    for i in indexes:
        rp[i:] += 360
        
    N = polynomial_sample_length
    N2 = N/2
    
    rp_fita = np.empty_like(rp)+np.nan
    rp_fitb = np.empty_like(rp)+np.nan
    
    for i in range(0,len(rp)+N, N):
        #indexes for subsets for overlapping a and b polynomials
        ia1 = max(0,i-N2)
        ia2 = min(len(rp),i+N2)
        ib1 = i
        ib2 = i+N
        
        #fit a polynomial
        if ia1<len(rp):
            z = np.polyfit(t[ia1:ia2]-t[ia1], rp[ia1:ia2], 3)
            rp_fita[ia1:ia2] = np.poly1d(z)(t[ia1:ia2]-t[ia1])
            
        if ib1<len(rp):
            #fit b polynomial
            z = np.polyfit(t[ib1:ib2]-t[ib1], rp[ib1:ib2], 3)
            rp_fitb[ib1:ib2] = np.poly1d(z)(t[ib1:ib2]-t[ib1])
    
    weighta = (np.cos(np.arange(len(rp))/N*2*np.pi))/2+.5
    weightb = (-np.cos(np.arange(len(rp))/N*2*np.pi))/2+.5
    
    return (rp_fita*weighta + rp_fitb*weightb)%360
    

def find_polynomial_sample_length(rotor_position, sample_frq, rotor_speed):
    from wetb.signal.filters import differentiation
    def err(i):
        rpm_pos = differentiation(fix_rotor_position(rotor_position, sample_frq, rotor_speed, i))%180 / 360 * sample_frq * 60
        return np.sum((rpm_pos - rotor_speed)**2)
    x_lst = np.arange(100,2000,100)
    res = [err(x) for x in x_lst]
    return x_lst[np.argmin(res)]
    