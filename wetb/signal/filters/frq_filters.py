'''
Created on 27. mar. 2017

@author: mmpe
'''
import numpy as np
from scipy import signal

def sine_generator(sample_frq, sine_frq, duration):
    """Create a sine signal for filter test
    
    Parameters
    ----------
    sample_frq : int, float
        Sample frequency of returned signal [Hz]
    sine_frq : int, float
        Frequency of sine signal [Hz]
    duration : int, float
        Duration of returned signal [s]
        
    Returns
    -------
    t,sig : ndarray, ndarray
        time (t) and sine signal (sig)
    
    Examples
    --------
    >>> sine_generator(10,1,2) 
    """
    T = duration
    nsamples = sample_frq * T
    w = 2. * np.pi * sine_frq
    t = np.linspace(0, T, nsamples, endpoint=False)
    sig = np.sin(w * t)
    return t, sig


def low_pass(x, sample_frq, cutoff_frq, order=5):
    """Low pass filter (butterworth)
    
    Parameters
    ----------
    x : array_like
        Input signal
    sample_frq : int, float
        Sample frequency [Hz]
    cutoff_frq : int, float
        Cut off frequency [Hz]
    order : int
        Order of the filter (1th order: 20db per decade, 2th order:)
    
    Returns
    -------
    y : ndarray
        Low pass filtered signal
    
    
    Examples
    --------
    >>> 
    """
    nyquist_frq = 0.5 * sample_frq
    normal_cutoff = cutoff_frq / nyquist_frq
    b,a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, x)

def high_pass(x, sample_frq, cutoff_frq, order=5):
    """Low pass filter (butterworth)
    
    Parameters
    ----------
    x : array_like
        Input signal
    sample_frq : int, float
        Sample frequency [Hz]
    cutoff_frq : int, float
        Cut off frequency [Hz]
    order : int
        Order of the filter (1th order: 20db per decade, 2th order:)
    
    Returns
    -------
    y : ndarray
        Low pass filtered signal
    
    
    Examples
    --------
    >>> 
    """
    nyquist_frq = 0.5 * sample_frq
    normal_cutoff = cutoff_frq / nyquist_frq
    b,a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, x)

def frequency_response(sample_frq, cutoff_frq, type, order, plt=None):
    """Frequency response of low/high pass filter (butterworth
    
    Parameters
    ----------
    sample_frq : int, float
        Sample frequency [Hz]
    cutoff_frq : int, float
        Cut off frequency [Hz]
    type : {'low','high'}
        Low or high pass filter
    order : int
        Order of the filter (1th order: 20db per decade, 2th order: 40db per decade)
    plt : pyplot, optional
        If specified, the frequency response is plotted
    
    Returns
    -------
    w,h : ndarray, ndarray
        Frequency (w) in Hz and filter response in db
    
    
    Examples
    --------
    >>>  
    """
    nyquist_frq = 0.5 * sample_frq
    normal_cutoff = cutoff_frq / nyquist_frq
    assert 0<normal_cutoff<1, "cutoff frequency must be <= nyquist frequency"
    b,a = signal.butter(order, cutoff_frq, btype=type, analog=True)
    w, h = signal.freqs(b, a)
    h_db = 20 * np.log10(abs(h))
    if plt:
        plt.plot(w, h_db, label='%d order filter response'%order)

        plt.legend(loc=0)
        
        title = 'Butterworth filter frequency response'
        if plt.axes().get_title()!=title:
            plt.title(title)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dB]')
            plt.margins(.1, .1)
            plt.xscale('log')
            
            plt.grid(which='both', axis='both')
            plt.axvline(cutoff_frq, color='green') # cutoff frequency
        
    return w,h_db