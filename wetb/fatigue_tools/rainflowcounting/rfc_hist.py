import numpy as np
def rfc_hist(sig_rf, nrbins=46):
    """
    Histogram of rainflow counted cycles
    ====================================

    hist, bin_edges, bin_avg = rfc_hist(sig, nrbins=46)

    Divide the rainflow counted cycles of a signal into equally spaced bins.

    Created on Wed Feb 16 16:53:18 2011
    @author: David Verelst
    Modified 10.10.2011 by Mads M Pedersen to elimintate __copy__ and __eq__

    Parameters
    ----------
    sig_rf : array-like
        As output by rfc_astm or rainflow

    nrbins : int, optional
        Divide the rainflow counted amplitudes in a number of equally spaced
        bins.

    Returns
    -------
    hist : array-like
        Counted rainflow cycles per bin, has nrbins elements

    bin_edges : array-like
        Edges of the bins, has nrbins+1 elements.

    bin_avg : array-like
        Average rainflow cycle amplitude per bin, has nrbins elements.
    """

    rf_half = sig_rf

    # the Matlab approach is to divide into 46 bins
    bin_edges = np.linspace(0, 1, num=nrbins + 1) * rf_half.max()
    hist = np.histogram(rf_half, bins=bin_edges)[0]
    # calculate the average per bin
    hist_sum = np.histogram(rf_half, weights=rf_half, bins=bin_edges)[0]
    # replace zeros with one, to avoid 0/0
    hist_ = hist.copy()
    hist_[(hist == 0).nonzero()] = 1.0
    # since the sum is also 0, the avg remains zero for those whos hist is zero
    bin_avg = hist_sum / hist_

    return hist, bin_edges, bin_avg