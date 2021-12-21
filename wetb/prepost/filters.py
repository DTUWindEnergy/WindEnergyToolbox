# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:14:02 2013

@author: dave
"""
import numpy as np
import scipy as sp

from wetb.prepost import DataChecks as chk
from wetb.prepost.misc import calc_sample_rate
from wetb.prepost import mplutils


class Filters(object):

    def __init__(self):
        pass


    def smooth(self, x, window_len=11, window='hanning'):
        """
        Smooth the data using a window with requested size
        ==================================================

        This method is based on the convolution of a scaled window with the
        signal. The signal is prepared by introducing reflected copies of the
        signal (with the window size) in both ends so that transient parts are
        minimized in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd
            integer
            window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman' flat window will produce a moving average
            smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
        numpy.convolve, scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array
        instead of a string

        SOURCE: http://www.scipy.org/Cookbook/SignalSmooth
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            msg = "Input vector needs to be bigger than window size."
            raise ValueError(msg)

        if window_len<3:
            return x

        windowlist = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        if not window in windowlist:
            msg = "Window should be 'flat', 'hanning', 'hamming', 'bartlett',"
            msg += " or 'blackman'"
            raise ValueError(msg)

        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')
        return y

    def butter(self, time, data, **kwargs):
        """
        Source:
        https://azitech.wordpress.com/2011/03/15/
        designing-a-butterworth-low-pass-filter-with-scipy/
        """

        sample_rate = kwargs.get('sample_rate', None)
        if not sample_rate:
            sample_rate = calc_sample_rate(time)

        # The cutoff frequency of the filter.
        cutoff_hz = kwargs.get('cutoff_hz', 1.0)

        # design filter
        norm_pass = cutoff_hz/(sample_rate/2.0)
        norm_stop = 1.5*norm_pass
        (N, Wn) = sp.signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2,
                                    gstop=30, analog=0)
        (b, a) = sp.signal.butter(N, Wn, btype='low', analog=0, output='ba')

        # filtered output
        #zi = signal.lfiltic(b, a, x[0:5], x[0:5])
        #(y, zi) = signal.lfilter(b, a, x, zi=zi)
        data_filt = sp.signal.lfilter(b, a, data)

        return data_filt

    def fir(self, time, data, **kwargs):
        """
        Based on the xxample from the SciPy cook boock, see
        http://www.scipy.org/Cookbook/FIRFilter

        Parameters
        ----------

        time : ndarray(n)

        data : ndarray(n)

        plot : boolean, default=False

        figpath : str, default=False

        figfile : str, default=False

        sample_rate : int, default=None
            If None, sample rate will be calculated from the given signal

        freq_trans_width : float, default=1
            The desired width of the transition from pass to stop,
            relative to the Nyquist rate.

        ripple_db : float, default=10
            The desired attenuation in the stop band, in dB.

        cutoff_hz : float, default=10
            Frequencies above cutoff_hz are filtered out

        Returns
        -------

        filtered_x : ndarray(n - (N-1))
            filtered signal

        N : float
            order of the firwin filter

        delay : float
            phase delay due to the filtering process

        """

        plot = kwargs.get('plot', False)
        figpath = kwargs.get('figpath', False)
        figfile = kwargs.get('figfile', False)

        sample_rate = kwargs.get('sample_rate', None)
        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        freq_trans_width = kwargs.get('freq_trans_width', 1)

        # The desired attenuation in the stop band, in dB.
        ripple_db = kwargs.get('ripple_db', 10)

        # The cutoff frequency of the filter.
        cutoff_hz = kwargs.get('cutoff_hz', 10)

        chk.array_1d(time)
        chk.array_1d(data)

        if not sample_rate:
            sample_rate = calc_sample_rate(time)

        #------------------------------------------------
        # Create a FIR filter and apply it to data[:,channel]
        #------------------------------------------------

        # The Nyquist rate of the signal.
        nyq_rate = sample_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = freq_trans_width/nyq_rate

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = sp.signal.kaiserord(ripple_db, width)

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = sp.signal.firwin(N, cutoff_hz/nyq_rate,
                                  window=('kaiser', beta))

        # Use lfilter to filter x with the FIR filter.
        filtered_x = sp.signal.lfilter(taps, 1.0, data)

        # The phase delay of the filtered signal.
        delay = 0.5 * (N-1) / sample_rate

#        # the filtered signal, shifted to compensate for the phase delay.
#        time_shifted = time-delay
#        # the "good" part of the filtered signal.  The first N-1
#        # samples are "corrupted" by the initial conditions.
#        time_good = time[N-1:] - delay

        if plot:
            self.plot_fir(figpath, figfile, time, data, filtered_x, N, delay,
                 sample_rate, taps, nyq_rate)

        return filtered_x, N, delay


    def plot_fir(self, figpath, figfile, time, data, filtered_x, N, delay,
                 sample_rate, taps, nyq_rate):
        """
        """

        #------------------------------------------------
        # Setup the figure parameters
        #------------------------------------------------

        plot = mplutils.A4Tuned()
        plot.setup(figpath+figfile, nr_plots=3, grandtitle=figfile,
                         figsize_y=20, wsleft_cm=2.)

        #------------------------------------------------
        # Plot the FIR filter coefficients.
        #------------------------------------------------
        plot_nr = 1
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)
        ax1.plot(taps, 'bo-', linewidth=2)
        ax1.set_title('Filter Coefficients (%d taps)' % N)
        ax1.grid(True)

        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------

        plot_nr += 1
        ax2 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)

        w, h = sp.signal.freqz(taps, worN=8000)
        ax2.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Gain')
        ax2.set_title('Frequency Response')
        ax2.set_ylim(-0.05, 1.05)
#        ax2.grid(True)

        # in order to place the nex axes inside following figure, first
        # determine the ax2 bounding box
        # points: a 2x2 numpy array of the form [[x0, y0], [x1, y1]]
        ax2box = ax2.get_window_extent().get_points()
        # seems to be expressed in pixels so convert to relative coordinates
#        print ax2box
        # figure size in pixels
        figsize_x_pix = plot.figsize_x*plot.dpi
        figsize_y_pix = plot.figsize_y*plot.dpi
        # ax2 box in relative coordinates
        ax2box[:,0] = ax2box[:,0] / figsize_x_pix
        ax2box[:,1] = ax2box[:,1] / figsize_y_pix
#        print ax2box[0,0], ax2box[1,0], ax2box[0,1], ax2box[1,1]
        # left position new box at 10% of x1
        left   = ax2box[0,0] + ((ax2box[1,0] - ax2box[0,0]) * 0.15)
        bottom = ax2box[0,1] + ((ax2box[1,1] - ax2box[0,1]) * 0.30)  # x2
        width  = (ax2box[1,0] - ax2box[0,0]) * 0.35
        height = (ax2box[1,1] - ax2box[0,1]) * 0.6
#        print [left, bottom, width, height]

        # left inset plot.
        # [left, bottom, width, height]
#        ax2a = plot.fig.add_axes([0.42, 0.6, .45, .25])
        ax2a = plot.fig.add_axes([left, bottom, width, height])
        ax2a.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2a.set_xlim(0,8.0)
        ax2a.set_ylim(0.9985, 1.001)
        ax2a.grid(True)

        # right inset plot
        left   = ax2box[0,0] + ((ax2box[1,0] - ax2box[0,0]) * 0.62)
        bottom = ax2box[0,1] + ((ax2box[1,1] - ax2box[0,1]) * 0.30)  # x2
        width  = (ax2box[1,0] - ax2box[0,0]) * 0.35
        height = (ax2box[1,1] - ax2box[0,1]) * 0.6

        # Lower inset plot
#        ax2b = plot.fig.add_axes([0.42, 0.25, .45, .25])
        ax2b = plot.fig.add_axes([left, bottom, width, height])
        ax2b.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2b.set_xlim(12.0, 20.0)
        ax2b.set_ylim(0.0, 0.0025)
        ax2b.grid(True)

        #------------------------------------------------
        # Plot the original and filtered signals.
        #------------------------------------------------

        # The phase delay of the filtered signal.
        delay = 0.5 * (N-1) / sample_rate

        plot_nr += 1
        ax3 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)
        # Plot the original signal.
        ax3.plot(time, data, label='original signal')
        # Plot the filtered signal, shifted to compensate for the phase delay.
        ax3.plot(time-delay, filtered_x, 'r-', label='filtered signal')
        # Plot just the "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        ax3.plot(time[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

        ax3.set_xlabel('t')
        ax3.grid(True)

        plot.save_fig()


    def scipy_example(self, time, data, figpath, sample_rate=None):
        """
        Example from the SciPy Cookboock, see
        http://www.scipy.org/Cookbook/FIRFilter
        """

        chk.array_1d(time)
        chk.array_1d(data)

        if not sample_rate:
            sample_rate = calc_sample_rate(time)

        #------------------------------------------------
        # Create a FIR filter and apply it to data[:,channel]
        #------------------------------------------------

        # The Nyquist rate of the signal.
        nyq_rate = sample_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 5.0/nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = sp.signal.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 10.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = sp.signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        # Use lfilter to filter x with the FIR filter.
        filtered_x = sp.signal.lfilter(taps, 1.0, data)

        #------------------------------------------------
        # Setup the figure parameters
        #------------------------------------------------
        figfile = 'filterdesign'

        plot = mplutils.A4Tuned()
        plot.setup(figpath+figfile, nr_plots=3, grandtitle=figfile,
                         figsize_y=20, wsleft_cm=2.)

        #------------------------------------------------
        # Plot the FIR filter coefficients.
        #------------------------------------------------
        plot_nr = 1
        ax1 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)
        ax1.plot(taps, 'bo-', linewidth=2)
        ax1.set_title('Filter Coefficients (%d taps)' % N)
        ax1.grid(True)

        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------

        plot_nr += 1
        ax2 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)

        w, h = sp.signal.freqz(taps, worN=8000)
        ax2.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Gain')
        ax2.set_title('Frequency Response')
        ax2.set_ylim(-0.05, 1.05)
#        ax2.grid(True)

        # in order to place the nex axes inside following figure, first
        # determine the ax2 bounding box
        # points: a 2x2 numpy array of the form [[x0, y0], [x1, y1]]
        ax2box = ax2.get_window_extent().get_points()
        # seems to be expressed in pixels so convert to relative coordinates
#        print ax2box
        # figure size in pixels
        figsize_x_pix = plot.figsize_x*plot.dpi
        figsize_y_pix = plot.figsize_y*plot.dpi
        # ax2 box in relative coordinates
        ax2box[:,0] = ax2box[:,0] / figsize_x_pix
        ax2box[:,1] = ax2box[:,1] / figsize_y_pix
#        print ax2box[0,0], ax2box[1,0], ax2box[0,1], ax2box[1,1]
        # left position new box at 10% of x1
        left   = ax2box[0,0] + ((ax2box[1,0] - ax2box[0,0]) * 0.15)
        bottom = ax2box[0,1] + ((ax2box[1,1] - ax2box[0,1]) * 0.30)  # x2
        width  = (ax2box[1,0] - ax2box[0,0]) * 0.35
        height = (ax2box[1,1] - ax2box[0,1]) * 0.6
#        print [left, bottom, width, height]

        # left inset plot.
        # [left, bottom, width, height]
#        ax2a = plot.fig.add_axes([0.42, 0.6, .45, .25])
        ax2a = plot.fig.add_axes([left, bottom, width, height])
        ax2a.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2a.set_xlim(0,8.0)
        ax2a.set_ylim(0.9985, 1.001)
        ax2a.grid(True)

        # right inset plot
        left   = ax2box[0,0] + ((ax2box[1,0] - ax2box[0,0]) * 0.62)
        bottom = ax2box[0,1] + ((ax2box[1,1] - ax2box[0,1]) * 0.30)  # x2
        width  = (ax2box[1,0] - ax2box[0,0]) * 0.35
        height = (ax2box[1,1] - ax2box[0,1]) * 0.6

        # Lower inset plot
#        ax2b = plot.fig.add_axes([0.42, 0.25, .45, .25])
        ax2b = plot.fig.add_axes([left, bottom, width, height])
        ax2b.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
        ax2b.set_xlim(12.0, 20.0)
        ax2b.set_ylim(0.0, 0.0025)
        ax2b.grid(True)

        #------------------------------------------------
        # Plot the original and filtered signals.
        #------------------------------------------------

        # The phase delay of the filtered signal.
        delay = 0.5 * (N-1) / sample_rate

        plot_nr += 1
        ax3 = plot.fig.add_subplot(plot.nr_rows, plot.nr_cols, plot_nr)
        # Plot the original signal.
        ax3.plot(time, data, label='original signal')
        # Plot the filtered signal, shifted to compensate for the phase delay.
        ax3.plot(time-delay, filtered_x, 'r-', label='filtered signal')
        # Plot just the "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        ax3.plot(time[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

        ax3.set_xlabel('t')
        ax3.grid(True)

        plot.save_fig()
