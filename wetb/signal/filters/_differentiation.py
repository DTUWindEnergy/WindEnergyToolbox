'''
Created on 29. mar. 2017

@author: mmpe
'''
from wetb.signal.filters.frq_filters import low_pass
import numpy as np
def differentiation(x,sample_frq=None, cutoff_frq=None):
    """Differentiate the signal
    
    Parameters
    ----------
    x : array_like
        The input signal
    sample_frq : int, float or None, optional
        sample frequency of signal (only required if low pass filer is applied)
    cutoff_frq : int, float or None, optional
        Low pass filter cut off (frequencies higher than this frequency will be suppressed)
    Returns
    -------
    y : ndarray
        differentiated signal
    
    
    Examples
    --------
    >>> differentiation([1,2,1,0,1,1])
    """ 
 
            
    if cutoff_frq is not None:
        assert sample_frq is not None, "Argument sample_frq must be set to apply low pass filter"
        x = low_pass(x, sample_frq, cutoff_frq)
    else:
        x = np.array(x) 
    dy = np.r_[x[1]-x[0], (x[2:]-x[:-2])/2, x[-1]-x[-2]]
    return dy