# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:15:25 2018

@author: shfe@dtu.dk

The model proposed by Leblanc
"""
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

class LeblancModel(object):
    
    
    def __init__(self, xi_b, xi_c, N):
        """
        Initialize the input parameters
        """
        
        self.xi_b = xi_b
        self.xi_c = xi_c
        self.N = N
        self.alpha = 0.31
        
    def coeff(self, rd = 0.38):
        """
        Calculate the parameter alpha
        """
        # Calculate the Tb, Tc
        if not 0 <= self.xi_b <= 1:
            raise ValueError('The maximum load ratio should between 0 to 1')
        
        if not -1 <= self.xi_c <= 1:
            raise ValueError('The cyclic load ratio should between -1 to 1')
        
        if rd == 0.38:
            if self.xi_b <= 0.051:
                Tb = 0
            else:
                Tb = 0.4238*self.xi_b - 0.0217
        
        elif rd == 0.04:
            if self.xi_b <= 0.1461:
                Tb = 0
            else:
                Tb = 0.3087*self.xi_b - 0.0451
                
        else:
            raise ValueError('The relative density value is not validated yet')
                
                
        if -1 <= self.xi_c < -0.65:
            Tc = 13.71*self.xi_c + 13.71
            
        elif -0.65 <= self.xi_c < 0:
            Tc = -5.54*self.xi_c + 1.2
            
        else:
            Tc = -1.2*self.xi_c + 1.2
        
        return Tb, Tc 
    
    def evoluation(self, y1):
        """
        Evoluation model for cyclic accumulated rotation
        """
        
        Tb, Tc = self.coeff()
        yn = y1 * Tb * Tc * self.N**0.31
        
        return yn
        
    def equivalentnum(self, y1, yn):
        """
        Equivalent number for cylces
        """
        Tb, Tc = self.coeff()
        num = (yn/(y1*Tb*Tc))**(1/self.alpha)
        
        return num

def superposition(xi_b, xi_c, N_cases, y1_cases, s0 = 0, y0 = 0):

    
    # Check the array equal number of items
    if not (np.size(xi_b) == np.size(xi_c) and np.size(xi_b) == np.size(N_cases)):
        raise ValueError('Size for input should be identical')
#    print(s0)
    dyn = np.zeros(np.size(xi_b))
    yn = np.zeros(np.size(xi_b))
    y_static = np.zeros(np.size(xi_b))
    for i, (b, c, N, y1) in enumerate(zip(xi_b, xi_c, N_cases, y1_cases)):
        
        if i==0:
            
            em = LeblancModel(b, c, N)
            Tb, Tc = em.coeff()
            if s0 == 0:
                dyn[i] = em.evoluation(y1)
            else:
                N_eq = (s0/(y1*Tb*Tc))**(1/em.alpha)
                if N_eq < 1e8:
                    dyn[i] = (y1*Tb*Tc)*(N+N_eq)**em.alpha
                else:
                    dyn[i] = s0
            
            y_static[i] = max(y1, y0)
                               
        else:
            em = LeblancModel(b, c, N)
            Tb, Tc = em.coeff()
            N_eq = (dyn[i-1]/(y1*Tb*Tc))**(1/em.alpha)
            if N_eq < 1e8:
                dyn[i] = (y1*Tb*Tc)*(N_eq + N)**em.alpha
            else:
                dyn[i] = dyn[i-1]
                
            y_static[i] = max(y_static[i-1], y1)
        
        yn[i] = dyn[i] + y_static[i]
        
    return dyn, y_static, yn
    
def coeff_super(xi_b, xi_c, N_cases):
    # Check the alpha value
    if not (np.size(xi_b) == np.size(xi_c) and np.size(xi_b) == np.size(N_cases)):
        raise ValueError('Size for input should be identical')
    
    Tb_list = np.zeros(np.size(xi_b))
    Tc_list = np.zeros(np.size(xi_c))
    for i, (b, c, N) in enumerate(zip(xi_b, xi_c, N_cases)):
        
        em = LeblancModel(b, c, N)
        Tb, Tc = em.coeff()
        Tb_list[i] = Tb
        Tc_list[i] = Tc
        
    return Tb_list, Tc_list
    
    
def moment_ult(L, D, gamma):
    """
    Ultimate static moment
    
    Parameters
    ----------------
    L -- Pile length
    D -- Pile diameter
    gamma -- Submerged unit weight
    
    Output
    ----------------
    m_ult -- Ultimate static moment
    """
    
    
    m_ult = L**3*D*gamma*0.6
    
    return m_ult
    
def rot_moment(m, L, D, gamma, pa = 101, rd  = 0.04):
    """
    Ultimate static moment
    
    Parameters
    ----------------
    L -- Pile length
    D -- Pile diameter
    gamma -- Submerged unit weight
    m -- Moment
    
    Output
    ----------------
    theta - Rotational angle in radians
    """
    
    if rd == 0.04:
        m_ult = L**3*D*gamma*0.6/1e3   # unit: MN
        theta_norm = 0.038 * (m/m_ult)**2.33
        theta = theta_norm*np.sqrt((L*gamma)/pa)
    
    else:
        m_ult = L**3*D*gamma*1.2/1e3    # unit: MN
        theta_norm = 0.042 * (m/m_ult)**1.92
        theta = theta_norm*np.sqrt((L*gamma)/pa)

    return theta

