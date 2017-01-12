import numpy as np
import warnings

def psd(data, sample_frq=1., no_segments=1, bin_size=1):
    """Map a signal to frequency domain (Power spectrum density)

    Parameters
    ----------
    data : array_like
        The time domain signal
    sample_frq : float, optional
        The sample frequency of the signal. Affects the frequency part of the result only\n
        Default is 1.
    no_segments : int, optional
        Remove noise by dividing the signal into no_segments bins and average the PSD of the subsets.\n
        As the time span of the bins are shorter, low frequencies are sacrificed.\n
        If e.g. 2, the signal is split into two parts and the result is the mean
        of the PSD for the first and the last part of the signal\n
        Default is 1

        Be carefull when using many segments, as the result may be corrupted as the function does not use any window function.
    bin_size : int, optional
        Smoothen the PSD by dividing the frequency list and PSD values into bins of size 'bin_size' and return the means of the bins

    Returns
    -------
    f : array_like
        Frequencies
    psd : array_like
        Power spectrum density

    Examples
    --------
    >>> ds = OpenHawc2("scripts/test.sel")
    >>> f, psd = PSD(ds(3)[:],ds.sample_frq, 2)
    >>> Plot(yscale='log')
    >>> PlotData(None, f, psd)
    """
    if no_segments is None:
        no_segments = 1

    nfft = data.shape[-1] // no_segments
    data = data[:no_segments * nfft].reshape(no_segments, nfft)


    NumUniquePts = int(np.ceil((nfft + 1) / 2))
    f = np.linspace(0, sample_frq / 2., NumUniquePts)
    fftx = np.abs(np.fft.rfft(data))

    fftx = fftx.sum(0)
    y = fftx / (nfft * no_segments)

    y = y ** 2
    y *= 2  # Because negative half is ignored

    y /= f[1] - f[0]  # distribute discrete energy values to continuous areas


    f, y = f[1:], y[1:]  # Remove 0Hz frequency contribution
    bins = (NumUniquePts - 1) // bin_size
    def smoothen(x, N):
        M = (x.shape[0] - N + 1) // N
        return np.array([x[i:M * N + i].reshape(M, N).mean(1) for i in range(N)]).T.flatten()
    if bin_size > 1:
        f, y = smoothen(f, bin_size), smoothen(y, bin_size)
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore", category=RuntimeWarning)  # ignore runtime warning of nanmean([nan])
#        f = np.nanmean(f[:bins * bin_size].reshape(bins, f.shape[0] // bins), 1)
#        y = np.nanmean(y[:bins * bin_size].reshape(bins, y.shape[0] // bins), 1)
    return (f, y)
