# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:22:50 2011

@author: dave
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import int
from builtins import dict
from builtins import round
from future import standard_library
standard_library.install_aliases()




# external libraries
import numpy as np

import matplotlib as mpl
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


def subplots(nrows=1, ncols=1, figsize=(12,8), dpi=120, num=0):
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

    fig = mpl.figure.Figure(figsize=figsize, dpi=dpi)
    canvas = FigCanvas(fig)
    fig.set_canvas(canvas)
    axes = np.ndarray((nrows, ncols), dtype=np.object)
    plt_nr = 1
    for row in range(nrows):
        for col in range(ncols):
            axes[row,col] = fig.add_subplot(nrows, ncols, plt_nr)
            plt_nr += 1
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


def p4psd(ax, rpm_mean, p_max=17, y_pos_rel=0.25, color='g', ls='--',
          col_text='w'):
    """
    Add the P's on a PSD

    fn_max is the maximum value on the plot (ax.xlim). This only works when
    setting the xlim of the plot before calling p4psd.

    Parameters
    ----------

    ax

    rpm_mean

    p_max : int, default=17

    y_pos_rel : int or list, default=0.25
    """
    if isinstance(y_pos_rel, float) or isinstance(y_pos_rel, int):
        y_pos_rel = [y_pos_rel]*p_max

    f_min = ax.get_xlim()[0]
    f_max = ax.get_xlim()[1]

    # add the P's
    bbox = dict(boxstyle="round", edgecolor=color, facecolor=color)
    for i, p in enumerate(range(1, p_max)):
        p_freq = p * rpm_mean / 60.0
        if p_freq > f_max:
            break
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


def peaks(ax, freqs, Pxx, fn_max, min_h, nr_peaks=15, col_line='k',
          ypos_mean=0.14, col_text='w', ypos_delta=0.06, bbox_alpha=0.5):
    """
    indicate the peaks
    """
    i_fn_max = np.abs(freqs - fn_max).argmin()
    # ignore everything above fn_max
    freqs = freqs[:i_fn_max]
    Pxx = Pxx[:i_fn_max]
    Pxx_log = 10.*np.log10(Pxx)
    try:
        pi = wafo.misc.findpeaks(Pxx_log, n=len(Pxx), min_h=min_h)
        print('len Pxx', len(Pxx_log), 'nr of peaks:', len(pi))
    except Exception as e:
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
    yrange_plot = Pxx_log.max() - Pxx_log.min()
    for peak_nr, ii in enumerate(pi):
        freq_peak = freqs[ii]
#        Pxx_peak = Pxx_log[ii]
        # take the average frequency and plot vertical line
        ax.axvline(x=freq_peak, linewidth=1, color=col_line, alpha=0.6)
        # and the value in a text box
        textbox = '%2.2f' % freq_peak
        if switch:
            # locate at the min value (down the plot), but a little
            # lower so it does not interfere with the plot itself
            # if ax.set_yscale('log') True, set log values as coordinates!
            text_ypos = Pxx_log.min() + yrange_plot*0.1
            text_ypos = ypos_mean + ypos_delta
            switch = False
        else:
            # put it a little lower than the max value so it does
            # not mess with the title (up the plot)
            text_ypos = Pxx_log.min() - yrange_plot*0.4
            text_ypos = ypos_mean - ypos_delta
            switch = True
#        print('%2.2e %2.2e %2.2e' % (yrange_plot, Pxx[:xlim].max(), Pxx[:xlim].min())
#        print peak, text_ypos
#        print textbox
#        print yrange_plot
        xrel = freq_peak/fn_max
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


def time_psd(results, labels, axes, alphas=[1.0, 0.7], colors=['k-', 'r-'],
             NFFT=None, res_param=250, f0=0, f1=None, nr_peaks=10, min_h=15,
             mark_peaks=False):
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
    ypos = [0.04, 0.90]

    for i, res in enumerate(results):
        time, data = res
        label = labels[i]
        col = colors[i]
        alpha = alphas[i]
        sps = int(round(1.0/np.diff(time).mean(), 0))
        if f1 is None:
            f1 = sps/2.0

        if NFFT is None:
            nfft = int(round(res_param * sps / (f1-f0), 0))
        elif isinstance(NFFT, list):
            nfft = NFFT[i]
        else:
            nfft = NFFT
        if nfft > len(data):
            nfft = len(data)

        # calculate the PSD
        Pxx, freqs = mpl.mlab.psd(data, NFFT=nfft, Fs=sps)

        i0 = np.abs(freqs - f0).argmin()
        i1 = np.abs(freqs - f1).argmin()

        # plotting psd, marking peaks
        axes[0].plot(freqs[i0:i1], Pxx[i0:i1], col, label=label, alpha=alpha)
        if mark_peaks:
            axes[0] = peaks(axes[0], freqs[i0:i1], Pxx[i0:i1], fn_max=f1,
                            nr_peaks=nr_peaks, col_line=col[:1],
                            ypos_delta=0.04, bbox_alpha=0.5, col_text='w',
                            ypos_mean=ypos[i], min_h=min_h)
        # plotting time series
        axes[1].plot(time, data, col, label=label, alpha=alpha)

    axes[0].set_yscale('log')
    axes[0].set_xlabel('frequency [Hz]')
    axes[1].set_xlabel('time [s]')
    for ax in axes:
        leg = ax.legend(loc='best', borderaxespad=0)
        # leg is None when no labels have been defined
        if leg is not None:
            leg.get_frame().set_alpha(0.7)
        ax.grid(True)

    return axes


if __name__ == '__main__':

    pass
