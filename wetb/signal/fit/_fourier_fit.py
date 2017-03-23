'''
Created on 07/07/2015

@author: MMPE
'''
import numpy as np
from wetb.signal.fit import bin_fit


def fourier_fit(y, max_nfft, x=None):
    """Approximate a signal, y, with Fourier fit"""
    d = np.arange(360)
    return d, lambda deg : np.interp(deg%360, d, F2x(x2F(y, max_nfft, x)))

def fourier_fit_old(y, nfft):
    F = np.zeros(len(y), dtype=np.complex)
    F[:nfft + 1] = x2F(y, nfft)[:nfft + 1]
    return np.fft.ifft(F) * len(F)

def F2x(F_coefficients):
    """Compute signal from Fourier coefficients"""
    F = np.zeros(360, dtype=np.complex)
    nfft = len(F_coefficients) // 2
    F[:nfft + 1] = np.conj(F_coefficients[:nfft + 1])
    F[1:nfft + 1] += (F_coefficients[-nfft:][::-1])
    return np.real(np.fft.ifft(F) * len(F))

def x2F(y, max_nfft, x=None):
    """Compute Fourier coefficients from signal (signal may contain NANs)"""
    d = np.arange(360)
    if x is not None:
        x,fit = bin_fit(x,y, d)
        y = fit(d)
        
    nfft = min(max_nfft, len(y) // 2 + 1)
    n = len(y)
    N = nfft * 2 + 1
    theta = np.linspace(0, 2 * np.pi, n + 1)[:n]
    theta[np.isnan(y)] = np.nan
    a = np.empty((nfft * 2 + 1, nfft * 2 + 1))
    b = np.empty(nfft * 2 + 1)
    A0_lst = lambda dF : 2 * np.nansum(1 * dF)
    A_lst = lambda dF : [2 * np.nansum(np.cos(i * theta) * dF) for i in range(1, nfft + 1)]
    B_lst = lambda dF : [2 * np.nansum(np.sin(i * theta) * dF) for i in range(1, nfft + 1)]
    row = lambda dF : np.r_[A0_lst(dF), A_lst(dF), B_lst(dF)]


    for i in range(nfft + 1):
        a[i, :] = row(np.cos(i * theta))
        b[i] = 2 * np.nansum(y * np.cos(i * theta))
    for i, r in enumerate(range(nfft + 1, nfft * 2 + 1), 1):
        a[r, :] = row(np.sin(i * theta))
        b[r] = 2 * np.nansum(y * np.sin(i * theta))
    AB = np.linalg.solve(a, b)

    F = np.zeros(n, dtype=np.complex)

    F = np.r_[AB[0], (AB[1:nfft + 1] + 1j * AB[nfft + 1:]), np.zeros(nfft) ]
    return F

def rx2F(y, max_nfft, x=None):
    """Convert non-complex signal, y, to single sided Fourier components, that satifies x(t) = sum(X(cos(iw)+sin(iw)), i=0..N)"""
    d = np.arange(360)
    if x is not None:
        x,fit = bin_fit(x,y, d)
        y = fit(d)
    F = np.fft.rfft(y) / len(y)
    F[1:-1] *= 2  # add negative side
    F = np.conj(F)
    return F[:max_nfft + 1]

def rF2x(rF):
    """Convert single sided Fourier components, that satisfies x(t) = sum(X(cos(iw)+sin(iw)), i=0..N) to non-complex signal"""
    rF = np.conj(rF)
    rF[1:] /= 2
    rF = np.r_[rF, np.zeros(181 - len(rF), dtype=np.complex)]
    return np.fft.irfft(rF) * 360
