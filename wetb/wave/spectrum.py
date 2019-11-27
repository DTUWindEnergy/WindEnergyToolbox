# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:33:21 2018

@author: shfe
"""

from __future__ import absolute_import, division

import warnings

from scipy.stats import moment
import scipy.integrate as integrate
import scipy.special as sp
import numpy as np
from numpy import (atleast_1d, minimum, maximum, ones_like,
                   exp, log, sqrt, where, pi, isfinite, cosh, zeros_like, flatnonzero)
                   
import matplotlib.pyplot as plt


def sech(x):
    return 1.0 / cosh(x)

class ModelSpectrum(object):
    type = 'ModelSpectrum'

    def __init__(self, Hm0=7.0, Tp=11.0, **kwds):
        self.Hm0 = Hm0
        self.Tp = Tp

    def chk_seastate(self):
        """ Check if seastate is valid
        """

        if self.Hm0 < 0:
            raise ValueError('Hm0 can not be negative!')

        if self.Tp <= 0:
            raise ValueError('Tp must be positve!')

        if self.Hm0 == 0.0:
            warnings.warn('Hm0 is zero!')

        self._chk_extra_param()

    def _chk_extra_param(self):
        pass
    
def jonswap_peakfact(Hm0, Tp):
    """ Jonswap peakedness factor, gamma, given Hm0 and Tp

    Parameters
    ----------
    Hm0 : significant wave height [m].
    Tp  : peak period [s]

    Returns
    -------
    gamma : Peakedness parameter of the JONSWAP spectrum

    Details
    -------
    A standard value for GAMMA is 3.3. However, a more correct approach is
    to relate GAMMA to Hm0 and Tp:
         D = 0.036-0.0056*Tp/sqrt(Hm0)
        gamma = exp(3.484*(1-0.1975*D*Tp**4/(Hm0**2)))
    This parameterization is based on qualitative considerations of deep water
    wave data from the North Sea, see Torsethaugen et. al. (1984)
    Here GAMMA is limited to 1..7.

    NOTE: The size of GAMMA is the common shape of Hm0 and Tp.

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import pylab as plb
    >>> Tp,Hs = plb.meshgrid(range(4,8),range(2,6))
    >>> gam = wsm.jonswap_peakfact(Hs,Tp)

    >>> Hm0 = plb.linspace(1,20)
    >>> Tp = Hm0
    >>> [T,H] = plb.meshgrid(Tp,Hm0)
    >>> gam = wsm.jonswap_peakfact(H,T)
    >>> v = plb.arange(0,8)

    >>> Hm0 = plb.arange(1,11)
    >>> Tp  = plb.linspace(2,16)
    >>> T,H = plb.meshgrid(Tp,Hm0)
    >>> gam = wsm.jonswap_peakfact(H,T)

    h = plb.contourf(Tp,Hm0,gam,v);h=plb.colorbar()
    h = plb.plot(Tp,gam.T)
    h = plb.xlabel('Tp [s]')
    h = plb.ylabel('Peakedness parameter')

    plb.close('all')

    See also
    --------
    jonswap
    """
    Hm0, Tp = atleast_1d(Hm0, Tp)

    x = Tp / sqrt(Hm0)

    gam = ones_like(x)

    k1 = flatnonzero(x <= 5.14285714285714)
    if k1.size > 0:  # limiting gamma to [1 7]
        xk = x.take(k1)
        D = 0.036 - 0.0056 * xk  # approx 5.061*Hm0**2/Tp**4*(1-0.287*log(gam))
        # gamma
        gam.put(k1, minimum(exp(3.484 * (1.0 - 0.1975 * D * xk ** 4.0)), 7.0))

    return gam


def jonswap_seastate(u10, fetch=150000., method='lewis', g=9.81,
                     output='dict'):
    """
    Return Jonswap seastate from windspeed and fetch

    Parameters
    ----------
    U10 : real scalar
        windspeed at 10 m above mean water surface [m/s]
    fetch : real scalar
        fetch [m]
    method : 'hasselman73' seastate according to Hasselman et. al. 1973
             'hasselman76' seastate according to Hasselman et. al. 1976
             'lewis'       seastate according to Lewis and Allos 1990
    g : real scalar
        accelaration of gravity [m/s**2]
    output : 'dict' or 'list'

    Returns
    -------
    seastate: dict  where
            Hm0    : significant wave height [m]
            Tp     : peak period [s]
            gamma  : jonswap peak enhancement factor.
            sigmaA,
            sigmaB : jonswap spectral width parameters.
            Ag     : jonswap alpha, normalization factor.

    Example
    --------
    >>> import wafo.spectrum.models as wsm
    >>> fetch = 10000; u10 = 10
    >>> ss = wsm.jonswap_seastate(u10, fetch, output='dict')
    >>> for key in sorted(ss.keys()): key, ss[key]
    ('Ag', 0.016257903375341734)
    ('Hm0', 0.51083679198275533)
    ('Tp', 2.7727680999585265)
    ('gamma', 2.4824142635861119)
    ('sigmaA', 0.07531733139517202)
    ('sigmaB', 0.09191208451225134)
    >>> S = wsm.Jonswap(**ss)
    >>> S.Hm0
    0.51083679198275533

    # Alternatively
    >>> ss1 = wsm.jonswap_seastate(u10, fetch, output='list')
    >>> S1 = wsm.Jonswap(*ss1)
    >>> S1.Hm0
    0.51083679198275533

    See also
    --------
    Jonswap


    References
    ----------
    Lewis, A. W. and Allos, R.N. (1990)
    JONSWAP's parameters: sorting out the inconscistencies.
    Ocean Engng, Vol 17, No 4, pp 409-415

    Hasselmann et al. (1973)
    Measurements of Wind-Wave Growth and Swell Decay during the Joint
    North Sea Project (JONSWAP).
    Ergansungsheft, Reihe A(8), Nr. 12, Deutschen Hydrografischen Zeitschrift.

    Hasselmann et al. (1976)
    A parametric wave prediction model.
    J. phys. oceanogr. Vol 6, pp 200-228

    """

    # The following formulas are from Lewis and Allos 1990:
    zeta = g * fetch / (u10 ** 2)  # dimensionless fetch, Table 1
    # zeta = min(zeta, 2.414655013429281e+004)
    if method.startswith('h'):
        if method[-1] == '3':  # Hasselman et.al (1973)
            A = 0.076 * zeta ** (-0.22)
            # dimensionless peakfrequency, Table 1
            ny = 3.5 * zeta ** (-0.33)
            # dimensionless surface variance, Table 1
            epsilon1 = 9.91e-8 * zeta ** 1.1
        else:  # Hasselman et.al (1976)
            A = 0.0662 * zeta ** (-0.2)
            ny = 2.84 * zeta ** (-0.3)  # dimensionless peakfrequency, Table 1
            # dimensionless surface variance, Eq.4
            epsilon1 = 1.6e-7 * zeta

        sa = 0.07
        sb = 0.09
        gam = 3.3
    else:
        A = 0.074 * zeta ** (-0.22)     # Eq. 10
        ny = 3.57 * zeta ** (-0.33)     # dimensionless peakfrequency, Eq. 11
        # dimensionless surface variance, Eq.12
        epsilon1 = 3.512e-4 * A * ny ** (-4.) * zeta ** (-0.1)
        sa = 0.05468 * ny ** (-0.32)      # Eq. 13
        sb = 0.078314 * ny ** (-0.16)     # Eq. 14
        gam = maximum(17.54 * zeta ** (-0.28384), 1)     # Eq. 15

    Tp = u10 / (ny * g)                          # Table 1
    Hm0 = 4 * sqrt(epsilon1) * u10 ** 2. / g            # Table 1
    if output[0] == 'l':
        return Hm0, Tp, gam, sa, sb, A
    else:
        return dict(Hm0=Hm0, Tp=Tp, gamma=gam, sigmaA=sa, sigmaB=sb, Ag=A)


def _gengamspec(wn, N=5, M=4):
    """ Return Generalized gamma spectrum in dimensionless form

    Parameters
    ----------
    wn : arraylike
        normalized frequencies, w/wp.
    N  : scalar
        defining the decay of the high frequency part.
    M  : scalar
        defining the spectral width around the peak.

    Returns
    -------
    S   : arraylike
        spectral values, same size as wn.

    The generalized gamma spectrum in non-
    dimensional form is defined as:

      S = G0.*wn.**(-N).*exp(-B*wn.**(-M))  for wn > 0
        = 0                              otherwise
    where
        B  = N/M
        C  = (N-1)/M
        G0 = B**C*M/gamma(C), Normalizing factor related to Bretschneider form

    Note that N = 5, M = 4 corresponds to a normalized
    Bretschneider spectrum.

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import numpy as np
    >>> wn = np.linspace(0,4,5)
    >>> wsm._gengamspec(wn, N=6, M=2)
    array([ 0.        ,  1.16765216,  0.17309961,  0.02305179,  0.00474686])

    See also
    --------
    Bretschneider
    Jonswap,
    Torsethaugen


    References
    ----------
    Torsethaugen, K. (2004)
    "Simplified Double Peak Spectral Model for Ocean Waves"
    In Proc. 14th ISOPE
    """
    w = atleast_1d(wn)
    S = zeros_like(w)

    k = flatnonzero(w > 0.0)
    if k.size > 0:
        B = N / M
        C = (N - 1.0) / M

        # A = Normalizing factor related to Bretschneider form
        # A = B**C*M/gamma(C)
        # S[k] = A*wn[k]**(-N)*exp(-B*wn[k]**(-M))
        logwn = log(w.take(k))
        logA = (C * log(B) + log(M) - sp.gammaln(C))
        S.put(k, exp(logA - N * logwn - B * exp(-M * logwn)))
    return S
    
class Jonswap(ModelSpectrum):

    """
    Jonswap spectral density model

    Member variables
    ----------------
    Hm0    : significant wave height (default 7 (m))
    Tp     : peak period             (default 11 (sec))
    gamma  : peakedness factor determines the concentraton
            of the spectrum on the peak frequency.
            Usually in the range  1 <= gamma <= 7.
            default depending on Hm0, Tp, see jonswap_peakedness)
    sigmaA : spectral width parameter for w<wp (default 0.07)
    sigmaB : spectral width parameter for w<wp (default 0.09)
    Ag     : normalization factor used when gamma>1:
    N      : scalar defining decay of high frequency part.   (default 5)
    M      : scalar defining spectral width around the peak. (default 4)
    method : String defining method used to estimate Ag when gamma>1
            'integration': Ag = 1/gaussq(Gf*ggamspec(wn,N,M),0,wnc) (default)
            'parametric' : Ag = (1+f1(N,M)*log(gamma)**f2(N,M))/gamma
            'custom'     : Ag = Ag
    wnc    : wc/wp normalized cut off frequency used when calculating Ag
                by integration (default 6)
    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    -----------
     The JONSWAP spectrum is defined as

             S(w) = A * Gf * G0 * wn**(-N)*exp(-N/(M*wn**M))
        where
             G0  = Normalizing factor related to Bretschneider form
             A   = Ag * (Hm0/4)**2 / wp     (Normalization factor)
             Gf  = j**exp(-.5*((wn-1)/s)**2) (Peak enhancement factor)
             wn  = w/wp
             wp  = angular peak frequency
             s   = sigmaA      for wn <= 1
                   sigmaB      for 1  <  wn
             j   = gamma,     (j=1, => Bretschneider spectrum)

    The JONSWAP spectrum is assumed to be especially suitable for the North
    Sea, and does not represent a fully developed sea. It is a reasonable model
    for wind generated sea when the seastate is in the so called JONSWAP range,
    i.e., 3.6*sqrt(Hm0) < Tp < 5*sqrt(Hm0)

    The relation between the peak period and mean zero-upcrossing period
    may be approximated by
             Tz = Tp/(1.30301-0.01698*gamma+0.12102/gamma)

    Examples
    ---------
    >>> import pylab as plb
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.Jonswap(Hm0=7, Tp=11,gamma=1)
    >>> S2 = wsm.Bretschneider(Hm0=7, Tp=11)
    >>> w = plb.linspace(0,5)
    >>> all(np.abs(S(w)-S2(w))<1.e-7)
    True

    h = plb.plot(w,S(w))
    plb.close('all')

    See also
    --------
     Bretschneider
     Tmaspec
     Torsethaugen

    References
    -----------
     Torsethaugen et al. (1984)
     Characteristica for extreme Sea States on the Norwegian continental shelf.
     Report No. STF60 A84123. Norwegian Hydrodyn. Lab., Trondheim

     Hasselmann et al. (1973)
     Measurements of Wind-Wave Growth and Swell Decay during the Joint
     North Sea Project (JONSWAP).
     Ergansungsheft, Reihe A(8), Nr. 12, Deutschen Hydrografischen Zeitschrift.
    """

    type = 'Jonswap'

    def __init__(self, Hm0=7.0, Tp=11.0, gamma=None, sigmaA=0.07, sigmaB=0.09,
                 Ag=None, N=5, M=4, method='integration', wnc=6.0,
                 chk_seastate=True):
        super(Jonswap, self).__init__(Hm0, Tp)
        self.N = N
        self.M = M
        self.sigmaA = sigmaA
        self.sigmaB = sigmaB
        self.gamma = gamma
        self.Ag = Ag
        self.method = method
        self.wnc = wnc

        if self.gamma is None or not isfinite(self.gamma) or self.gamma < 1:
            self.gamma = jonswap_peakfact(Hm0, Tp)

        self._pre_calculate_ag()

        if chk_seastate:
            self.chk_seastate()

    def _chk_extra_param(self):
        Tp = self.Tp
        Hm0 = self.Hm0
        gam = self.gamma
        outsideJonswapRange = Tp > 5 * sqrt(Hm0) or Tp < 3.6 * sqrt(Hm0)
        if outsideJonswapRange:
            txt0 = """
            Hm0=%g,Tp=%g is outside the JONSWAP range.
            The validity of the spectral density is questionable.
            """ % (Hm0, Tp)
            warnings.warn(txt0)

        if gam < 1 or 7 < gam:
            txt = """
            The peakedness factor, gamma, is possibly too large.
            The validity of the spectral density is questionable.
            """
            warnings.warn(txt)

    def _localspec(self, wn):
        Gf = self.peak_e_factor(wn)
        return Gf * _gengamspec(wn, self.N, self.M)

    def _check_parametric_ag(self, N, M, gammai):
        parameters_ok = 3 <= N <= 50 or 2 <= M <= 9.5 and 1 <= gammai <= 20
        if not parameters_ok:
            raise ValueError('Not knowing the normalization because N, ' +
                             'M or peakedness parameter is out of bounds!')
        if self.sigmaA != 0.07 or self.sigmaB != 0.09:
            warnings.warn('Use integration to calculate Ag when ' + 'sigmaA!=0.07 or sigmaB!=0.09')

    def _parametric_ag(self):
        """
        Original normalization

        NOTE: that  Hm0**2/16 generally is not equal to intS(w)dw
              with this definition of Ag if sa or sb are changed from the
              default values
        """
        self.method = 'parametric'

        N = self.N
        M = self.M
        gammai = self.gamma
        f1NM = 4.1 * (N - 2 * M ** 0.28 + 5.3) ** (-1.45 * M ** 0.1 + 0.96)
        f2NM = ((2.2 * M ** (-3.3) + 0.57) * N ** (-0.58 * M ** 0.37 + 0.53) -
                1.04 * M ** (-1.9) + 0.94)
        self.Ag = (1 + f1NM * log(gammai) ** f2NM) / gammai
        # if N == 5 && M == 4,
        #     options.Ag = (1+1.0*log(gammai).**1.16)/gammai
        #     options.Ag = (1-0.287*log(gammai))
        #     options.normalizeMethod = 'Three'
        # elseif  N == 4 && M == 4,
        #     options.Ag = (1+1.1*log(gammai).**1.19)/gammai

        self._check_parametric_ag(N, M, gammai)

    def _custom_ag(self):
        self.method = 'custom'
        if self.Ag <= 0:
            raise ValueError('Ag must be larger than 0!')

    def _integrate_ag(self):
        # normalizing by integration
        self.method = 'integration'
        if self.wnc < 1.0:
            raise ValueError('Normalized cutoff frequency, wnc, ' +
                             'must be larger than one!')
        area1, unused_err1 = integrate.quad(self._localspec, 0, 1)
        area2, unused_err2 = integrate.quad(self._localspec, 1, self.wnc)
        area = area1 + area2
        self.Ag = 1.0 / area

    def _pre_calculate_ag(self):
        """ PRECALCULATEAG Precalculate normalization.
        """
        if self.gamma == 1:
            self.Ag = 1.0
            self.method = 'parametric'
        elif self.Ag is not None:
            self._custom_ag()
        else:
            norm_ag = dict(i=self._integrate_ag,
                           p=self._parametric_ag,
                           c=self._custom_ag)[self.method[0]]
            norm_ag()

    def peak_e_factor(self, wn):
        """ PEAKENHANCEMENTFACTOR
        """
        w = maximum(atleast_1d(wn), 0.0)
        sab = where(w > 1, self.sigmaB, self.sigmaA)

        wnm12 = 0.5 * ((w - 1.0) / sab) ** 2.0
        Gf = self.gamma ** (exp(-wnm12))
        return Gf

    def __call__(self, wi):
        """ JONSWAP spectral density
        """
        w = atleast_1d(wi)
        if (self.Hm0 > 0.0):

            N = self.N
            M = self.M
            wp = 2 * pi / self.Tp
            wn = w / wp
            Ag = self.Ag
            Hm0 = self.Hm0
            Gf = self.peak_e_factor(wn)
            S = ((Hm0 / 4.0) ** 2 / wp * Ag) * Gf * _gengamspec(wn, N, M)
        else:
            S = zeros_like(w)
        return S

if __name__ == '__main__':
    plt.close('all')
    Hm0 = 0.8
    Tp = 4.0
    S = Jonswap(Hm0=Hm0, Tp=Tp, gamma = 3.3)
    w = np.arange(0, 50, 0.005)
    f = w / (2 * pi)
    Sw = S(w)
    Sf = Sw * 2 * pi
    plt.figure(figsize = (6, 6))
    plt.plot(f, Sf, lw = 2) # change to Hz
    plt.xlabel(r'$f$ $[Hz]$', fontsize = 20) 
    plt.ylabel(r'$S$', fontsize = 20)
    plt.xlim([0, 1])
    plt.grid()
    plt.tight_layout()