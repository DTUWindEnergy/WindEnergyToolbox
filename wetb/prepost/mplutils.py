# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:22:50 2011

@author: dave
"""
# external libraries
import numpy as np

import matplotlib as mpl
from matplotlib.figure import Figure
# use a headless backend
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
# wafo is an optional dependency only required for non default PSD peak marking
try:
    import wafo
except ImportError:
    pass


def make_fig(nrows=1, ncols=1, figsize=(12,8), dpi=120):
    """

    Equivalent function of pyplot.subplots(). The difference is that this one
    is not interactive and is used with backend plotting only.

    Parameters
    ----------
    nrows=1, ncols=1, figsize=(12,8), dpi=120

    Returns
    -------
    fig, canvas, axes


    """
    return subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)


def subplots(nrows=1, ncols=1, figsize=(12,8), dpi=120, num=0, subplot_kw={},
             gridspec_kw={}):
    """

    Equivalent function of pyplot.subplots(). The difference is that this one
    is not interactive and is used with backend plotting only.

    Parameters
    ----------
    nrows=1, ncols=1, figsize=(12,8), dpi=120

    num : dummy variable for compatibility

    Returns
    -------
    fig, axes


    """

    fig = Figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=nrows, ncols=ncols,
                        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    try:
        axes = axes.reshape((nrows,ncols))
    except AttributeError:
        pass
    # canvas = FigCanvas(fig)
    # fig.set_canvas(canvas)
    # axes = np.ndarray((nrows, ncols), dtype=object)
    # plt_nr = 1
    # for row in range(nrows):
    #     for col in range(ncols):
    #         axes[row,col] = fig.add_subplot(nrows, ncols, plt_nr, **subplot_kw)
    #         plt_nr += 1
    return fig, axes


def match_axis_ticks(ax1, ax2, ax1_format=None, ax2_format=None):
    """
    Match ticks of ax2 to ax1
    =========================

    ax1_format: '%1.1f'

    Parameters
    ----------

    ax1, ax2

    Returns
    -------

    ax1, ax2

    """
    # match the ticks of ax2 to ax1
    yticks1 = len(ax1.get_yticks())
    ylim2 = ax2.get_ylim()
    yticks2 = np.linspace(ylim2[0], ylim2[1], num=yticks1).tolist()
    ax2.yaxis.set_ticks(yticks2)

    # give the tick labels a given precision
    if ax1_format:
        majorFormatter = mpl.ticker.FormatStrFormatter(ax1_format)
        ax1.yaxis.set_major_formatter(majorFormatter)

    if ax2_format:
        majorFormatter = mpl.ticker.FormatStrFormatter(ax2_format)
        ax2.yaxis.set_major_formatter(majorFormatter)

    return ax1, ax2


def one_legend(*args, **kwargs):
    """First list all the axes as arguments. Any keyword arguments will be
    passed on to ax.legend(). Legend will be placed on the last axes that was
    passed as an argument.

    Parameters
    ----------

    Returns
    -------

    legend

    """
    # or more general: not only simple line plots (bars, hist, ...)
    objs = []
    for ax in args:
        objs += ax.get_legend_handles_labels()[0]
#    objs = [ax.get_legend_handles_labels()[0] for ax in args]
    labels = [obj.get_label() for obj in objs]
    # place the legend on the last axes
    leg = ax.legend(objs, labels, **kwargs)
    return leg


def p4psd(ax, rpm_mean, fmax=10, y_pos_rel=0.25, color='g', ls='--', ps=None,
          col_text='w'):
    """
    Add the P's on a PSD

    fmax is the maximum value on the plot (ax.xlim). This only works when
    setting the xlim of the plot before calling p4psd.

    Parameters
    ----------

    ax

    rpm_mean

    fmax : int, default=17
        stop plotting p's after fmax (when ps is None)

    ps : iterable of ints, default=None
        specify which p's to plot (ignores fmax)

    y_pos_rel : int or list, default=0.25
    """

    if ps is None:
        pmax = int(60*fmax/rpm_mean)
        ps = list(range(1, pmax))
    else:
        pmax = len(ps)

    if isinstance(y_pos_rel, float) or isinstance(y_pos_rel, int):
        y_pos_rel = [y_pos_rel]*pmax

    f_min = ax.get_xlim()[0]
    f_max = ax.get_xlim()[1]

    # add the P's
    bbox = dict(boxstyle="round", edgecolor=color, facecolor=color)
    for i, p in enumerate(ps):
        p_freq = p * rpm_mean / 60.0
        if p%3 == 0:
            alpha=0.5
            ax.axvline(x=p_freq, linewidth=1, color=color, alpha=0.7, ls=ls)
        else:
            alpha = 0.2
            ax.axvline(x=p_freq, linewidth=1, color=color, alpha=0.7, ls=ls)

        x = (p_freq - f_min) / (f_max - f_min)
        y = y_pos_rel[i]

        p_str = '%iP' % p
        bbox['alpha'] = alpha
        ax.text(x, y, p_str, fontsize=8, verticalalignment='bottom',
                horizontalalignment='center', bbox=bbox, color=col_text,
                transform=ax.transAxes)

    return ax


def peaks(ax, freqs, Pxx, fn_max, min_h, nr_peaks=15, min_p=0, col_line='k',
          ypos_mean=0.14, col_text='w', ypos_delta=0.06, bbox_alpha=0.5,
          verbose=False, format_text='%2.2f', period=False):
    """
    indicate the peaks

    Parameters
    ----------

    period : boolean, default=False
        If True, the period instead of the frequency is given in the text

    min_h : float
        The threshold in the rainflowfilter (default 0.05*range(S(:))).
        A zero value will return all the peaks of S.

    min_p : float, 0..1
        Only the peaks that are higher than min_p*max(max(S))
        min_p*(the largest peak in S) are returned (default  0).
    """
    i_fn_max = np.abs(freqs - fn_max).argmin()
    # ignore everything above fn_max
    freqs = freqs[:i_fn_max]
    Pxx = Pxx[:i_fn_max]
    Pxx_log = 10.*np.log10(Pxx)
    try:
        pi = wafo.misc.findpeaks(Pxx_log, n=len(Pxx), min_h=min_h, min_p=min_p)
        if verbose:
            print('len Pxx', len(Pxx_log), 'nr of peaks:', len(pi))
    except Exception as e:
        if verbose:
            print('len Pxx', len(Pxx_log))
            print('*** wafo.misc.findpeaks FAILED ***')
            print(e)
        return ax

    # only take the nr_peaks most significant heights
    pi = pi[:nr_peaks]
    # and sort them accoriding to frequency (or index)
    pi.sort()

    # mark the peaks with a circle
#    ax.plot(freqs[pi], Pxx[:xlim][pi], 'o')
    # and mark all peaks
    switch = True
    # yrange_plot = Pxx_log.max() - Pxx_log.min()
    for peak_nr, ii in enumerate(pi):
        freq_peak = freqs[ii]
#        Pxx_peak = Pxx_log[ii]
        # take the average frequency and plot vertical line
        ax.axvline(x=freq_peak, linewidth=1, color=col_line, alpha=0.6)
        if switch:
            # locate at the min value (down the plot), but a little
            # lower so it does not interfere with the plot itself
            # if ax.set_yscale('log') True, set log values as coordinates!
            # text_ypos = Pxx_log.min() + yrange_plot*0.1
            text_ypos = ypos_mean + ypos_delta
            switch = False
        else:
            # put it a little lower than the max value so it does
            # not mess with the title (up the plot)
            # text_ypos = Pxx_log.min() - yrange_plot*0.4
            text_ypos = ypos_mean - ypos_delta
            switch = True
#        print('%2.2e %2.2e %2.2e' % (yrange_plot, Pxx[:xlim].max(), Pxx[:xlim].min())
#        print peak, text_ypos
#        print textbox
#        print yrange_plot
        xrel = freq_peak/fn_max
        # and the value in a text box
        if period:
            freq_peak = 1/freq_peak
        textbox = format_text % freq_peak
        ax.text(xrel, text_ypos, textbox, fontsize=10, transform=ax.transAxes,
                 va='bottom', color=col_text, bbox=dict(boxstyle="round",
                 ec=col_line, fc=col_line, alpha=bbox_alpha,))

    return ax


def match_yticks(ax1, ax2, nr_ticks_forced=None, extend=False):
    """
    """

    if nr_ticks_forced is None:
        nr_yticks1 = len(ax1.get_yticks())
    else:
        nr_yticks1 = nr_ticks_forced
        ylim1 = ax1.get_ylim()
        yticks1 = np.linspace(ylim1[0], ylim1[1], num=nr_yticks1).tolist()
        ax1.yaxis.set_ticks(yticks1)

    ylim2 = ax2.get_ylim()
    yticks2 = np.linspace(ylim2[0], ylim2[1], num=nr_yticks1).tolist()
    ax2.yaxis.set_ticks(yticks2)

    if extend:
        offset1 = (ylim1[1] - ylim1[0])*0.1
        ax1.set_ylim(ylim1[0]-offset1, ylim1[1]+offset1)
        offset2 = (ylim2[1] - ylim2[0])*0.1
        ax2.set_ylim(ylim2[0]-offset2, ylim2[1]+offset2)

    return ax1, ax2


def psd(ax, time, sig, nfft=None, res_param=250, f0=0, f1=None, nr_peaks=10,
        min_h=15, min_p=0, mark_peaks=False, col='r-', label=None, alpha=1.0,
        ypos_peaks=0.9, ypos_peaks_delta=0.12, format_text='%2.2f', period=False):
    """Only plot the psd on a given axis and optionally mark the peaks.
    """

    sps = int(round(1.0/np.diff(time).mean(), 0))
    if f1 is None:
        f1 = sps/2.0

    if nfft is None:
        nfft = int(round(res_param * sps / (f1-f0), 0))
    if nfft > len(sig):
        nfft = len(sig)

    # calculate the PSD

    Pxx, freqs = mpl.mlab.psd(sig, NFFT=nfft, Fs=sps) # window=None
    # for clean signals (steps, sinus) use another window
    # for stochastic signal, the default window Hanning is good
    # in Scipy you can provide strings for the different windows, not so in mlab

    i0 = np.abs(freqs - f0).argmin()
    i1 = np.abs(freqs - f1).argmin()

    # plotting psd, marking peaks
    ax.plot(freqs[i0:i1], Pxx[i0:i1], col, label=label, alpha=alpha)
    if mark_peaks:
        ax = peaks(ax, freqs[i0:i1], Pxx[i0:i1], fn_max=f1, min_p=min_p,
                   nr_peaks=nr_peaks, col_line=col[:1], format_text=format_text,
                   ypos_delta=ypos_peaks_delta, bbox_alpha=0.5, period=period,
                   ypos_mean=ypos_peaks, min_h=min_h, col_text='w')

    return ax


def time_psd(results, labels, axes, alphas=[1.0, 0.7], colors=['k-', 'r-'],
             NFFT=None, res_param=250, f0=0, f1=None, nr_peaks=10, min_h=15,
             mark_peaks=False, xlabels=['frequency [Hz]', 'time [s]'],
             ypos_peaks=[0.3, 0.9], ypos_peaks_delta=[0.12, 0.12]):
    """
    Plot time series and the corresponding PSD of the channel.

    The frequency range depends on the sample rate: fn_max = sps/2.
    The number of points is: NFFT/2
    Consequently, the frequency resolution is: NFFT/SPS (points/Hz)

    len(data) > NFFT otherwise the results do not make any sense

    res_param = NFFT*frequency_range*sps
    good range for nondim_resolution is: 200-300

    Frequency range is based on f0 and f1

    for the PSD powers of 2 are faster, but the difference in performance with
    any other random number fluctuates on average and is not that relevant.

    Requires two axes!

    With the PSD peaks this only works nicely for 2 results

    Parameters
    ----------

    results : list
        list of time,data pairs (ndarrays)
    """

    axes = axes.ravel()

    for i, res in enumerate(results):
        time, data = res
        label = labels[i]
        col = colors[i]
        alpha = alphas[i]
        if isinstance(NFFT, list):
            nfft = NFFT[i]
        else:
            nfft = NFFT
        axes[0] = psd(axes[0], time, data, nfft=nfft, res_param=res_param,
                      f0=f0, f1=f1, nr_peaks=nr_peaks, min_h=min_h,
                      mark_peaks=mark_peaks, col=col, label=label, alpha=alpha,
                      ypos_peaks_delta=ypos_peaks_delta[i],
                      ypos_peaks=ypos_peaks[i])

        # plotting time series
        axes[1].plot(time, data, col, label=label, alpha=alpha)

    axes[0].set_yscale('log')
    if isinstance(xlabels, list):
        axes[0].set_xlabel(xlabels[0])
        axes[1].set_xlabel(xlabels[1])
    for ax in axes:
        leg = ax.legend(loc='best', borderaxespad=0)
        # leg is None when no labels have been defined
        if leg is not None:
            leg.get_frame().set_alpha(0.7)
        ax.grid(True)

    return axes


def get_list_colors(nr, cmap='magma'):
    """Returns a list of color tuples

    Paramters
    ---------

    nr : int
        number of colors

    cmap : str
        a matplotlib color map name, for example: viridis, plasma, magma, hot,
        cool, binary, see also https://matplotlib.org/users/colormaps.html

    Returns
    -------

    clist : list
        List continaing 'nr' color tuples (with 3 elements).

    """
    # define the number of positions you want to have the color for
    # select a color map
    cmap = mpl.cm.get_cmap(cmap, nr)
    # other color maps: 	cubehelix, afmhot, hot
    # convert to array
    cmap_arr = cmap(np.arange(nr))
    # and now you have each color as an RGB tuple as
    clist = []
    for i in cmap_arr:
        clist.append(tuple(i[0:3]))

    return clist


if __name__ == '__main__':

    pass
