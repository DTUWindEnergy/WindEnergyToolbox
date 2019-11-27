# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:34:12 2018

@author: shfe

Note:
This script is used for Klinkvort model used for cyclic accumulation calculation
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

class KlinkvortModel(object):
    
    
    def __init__(self, xi_b, xi_c, N):
        """
        Initialize the input parameters
        """
        
        self.xi_b = xi_b
        self.xi_c = xi_c
        self.N = N
        
    def alpha(self):
        """
        Calculate the parameter alpha
        """
        # Calculate the Tb, Tc
        if not 0 <= self.xi_b <= 1:
            raise ValueError('The maximum load ratio should between 0 to 1')
        
        if not -1 <= self.xi_c <= 1:
            raise ValueError('The cyclic load ratio should between -1 to 1')
            
        if self.xi_b <= 0.022:
            Tb = 0
        else:
            Tb = 0.61*self.xi_b - 0.013
        
        Tc = (self.xi_c + 0.63)*(self.xi_c - 1)*(self.xi_c - 1.64)
        alpha = Tb * Tc
        
        return Tb, Tc, alpha 
    
    def evoluation(self, y1):
        """
        Evoluation model for cyclic accumulated rotation
        """
        
        _, _, alpha = self.alpha()
        yn = y1 * self.N**alpha
        
        return yn
        
    def equivalentnum(self, y1, yn, alpha):
        """
        Equivalent number for cylces
        """
        num = (yn/y1)**(1/alpha)
        
        return num
        
def superposition(xi_b, xi_c, N_cases, y1_cases, s0 = 0):

    
    # Check the array equal number of items
    if not (np.size(xi_b) == np.size(xi_c) and np.size(xi_b) == np.size(N_cases)):
        raise ValueError('Size for input should be identical')
    print(s0)
    yn = np.zeros(np.size(xi_b))
    for i, (b, c, N, y1) in enumerate(zip(xi_b, xi_c, N_cases, y1_cases)):
        
        if i==0:
            
            em = KlinkvortModel(b, c, N)
            _, _, alpha = em.alpha()
            if s0 == 0:
                yn[i] = em.evoluation(y1)
            else:
                N_eq = (s0/y1)**(1/alpha)
                if N_eq < 1e8:
                    
#                    N_eq = (s0/y1)**(1/alpha)
                    yn[i] = y1*(N+N_eq)**alpha
                else:
                    yn[i] = s0
                        
                
        
        else:
            em = KlinkvortModel(b, c, N)
            _, _, alpha = em.alpha()
            N_eq = (yn[i-1]/y1)**(1/alpha)
            if N_eq < 1e8:
#                N_eq = (yn[i-1]/y1)**(1/alpha)
                yn[i] = y1*(N_eq + N)**alpha
            else:
                yn[i] = yn[i-1]

    return yn
    
def alpha_super(xi_b, xi_c, N_cases):
    # Check the alpha value
    if not (np.size(xi_b) == np.size(xi_c) and np.size(xi_b) == np.size(N_cases)):
        raise ValueError('Size for input should be identical')
    
    alpha_list = np.zeros(np.size(xi_b))
    for i, (b, c, N) in enumerate(zip(xi_b, xi_c, N_cases)):
        
        em = KlinkvortModel(b, c, N)
        _, _, alpha = em.alpha()
        alpha_list[i] = alpha
        
    return alpha_list
    
    
    
    