'''
Created on 22. mar. 2017

@author: mmpe
'''
import numpy as np
def linear_fit(x,y):
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return np.array([x.min(), x.max()]), lambda x : np.array(x)*slope+intercept , (slope, intercept)