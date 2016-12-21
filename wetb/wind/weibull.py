"""
Contents
--------
- `fit <#wetb.wind.weibull.fit>`_: Fit a weibull distribution, in terms of the parameters k and A, to the provided wind speeds
- `pdf <#wetb.wind.weibull.pdf>`_: Create Weibull pdf function
- `random <#wetb.wind.weibull.random>`_: Create a list of n random Weibull distributed values
"""
import numpy as np
import math
gamma = math.gamma
def pdf(A, k):
    """Create Weibull pdf function

    Parameters
    ----------
    A : float
        Scale parameter
    k : float
        Shape parameter

    Returns
    -------
    pdf-function

    Examples
    --------
    >>> from wetb.wind import weibull
    >>> pdf_func = weibull.pdf(3,4)
    >>> pdf_func(5)
    0.131007116969
    >>> from pylab import arange, plot, show
    >>> wsp = arange(20)
    >>> plot(wsp, pdf_func(wsp))
    >>> show()

    """

    return lambda x: k * x ** (k - 1) / A ** k * np.exp(-(x / A) ** k)

def cdf(A,k):
    return lambda x: 1-np.exp(-(x/A)**k)

def random(A, k, n):
    """Create a list of n random Weibull distributed values

    Parameters
    ----------
    A : float
        Scale parameter
    k : float
        Shape parameter
    n : int
        Number of values

    Returns
    -------
    x : array_like, shape (n,)
        n random Weibull distributed values

    Examples
    --------
    >>> from wetb.wind import weibull
    >>> from pylab import hist, show
    >>> hist(weibull.random(4,2,1000), 20)
    >>> show()
    """
    return A * np.random.weibull(k, n)

def fit(wsp):
    """Fit a weibull distribution, in terms of the parameters k and A, to the provided wind speeds

    Parameters
    ----------
    wsp : array_like
        Wind speeds

    Returns
    -------
    A : float
        Scale parameter
    k : float
        Shape parameter

    Examples
    --------
    >>> from wetb.wind import weibull
    >>> A,k = weibull.fit(wsp_lst)
    """
    res_pr_ms = 2  # number of wind speed bins pr m/s

    pdf, x = np.histogram(wsp, bins=np.arange(0, np.ceil(np.nanmax(wsp)), 1 / res_pr_ms))
    x = (x[1:] + x[:-1]) / 2
    N = np.sum(~np.isnan(wsp))
    pdf = pdf / N * res_pr_ms

    m = lambda n : np.sum(pdf * x ** n / res_pr_ms)
    from scipy.optimize import newton
    func = lambda k : gamma(1 / k + 1) ** 2 / gamma(2 / k + 1) - m(1) ** 2 / m(2)
    k = newton(func, 1)
    A = m(1) / gamma(1 / k + 1)
    return A, k

if __name__ == "__main__":
    from wetb.wind import weibull
    from pylab import hist, show, plot
    hist(weibull.random(10, 2, 10000), 20, normed=True)
    wsp = np.arange(0, 20, .5)
    plot(wsp, weibull.pdf(10, 2)(wsp))
    show()
