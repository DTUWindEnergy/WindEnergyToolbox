# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:21:11 2014

@author: dave
"""
import os
# import socket
import gc

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
#from scipy import interpolate as interp
#from scipy.optimize import fmin_slsqp
#from scipy.optimize import minimize
#from scipy.interpolate import interp1d
#import scipy.integrate as integrate
#http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
import pandas as pd

#import openpyxl as px
#import numpy as np

#import windIO
from wetb.prepost import mplutils
from wetb.prepost import Simulations as sim
from wetb.prepost import dlcdefs

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
# do not use tex on Gorm and or Jess
#if not socket.gethostname()[:2] in ['g-', 'je', 'j-']:
#    plt.rc('text', usetex=True)
plt.rc('legend', fontsize=11)
plt.rc('legend', numpoints=1)
plt.rc('legend', borderaxespad=0)


def merge_sim_ids(sim_ids, post_dirs, post_dir_save=False, columns=None):
    """
    """

    cols_extra = ['[run_dir]', '[res_dir]', '[wdir]', '[DLC]', '[Case folder]']
    min_itemsize={'channel':100, '[run_dir]':100, '[res_dir]':100, '[DLC]':10,
                  '[Case folder]':100}

    # map the run_dir to the same order as the post_dirs, labels
    run_dirs = []
    # avoid saving merged cases if there is only one!
    if type(sim_ids).__name__ == 'list' and len(sim_ids) == 1:
        sim_ids = sim_ids[0]

    # if sim_id is a list, combine the two dataframes into one
    df_stats = pd.DataFrame()
    if type(sim_ids).__name__ == 'list':
        for ii, sim_id in enumerate(sim_ids):
            if isinstance(post_dirs, list):
                post_dir = post_dirs[ii]
            else:
                post_dir = post_dirs
            cc = sim.Cases(post_dir, sim_id, rem_failed=True)
            df_stats, _, _ = cc.load_stats(leq=False)
            if columns is not None:
                df_stats = df_stats[columns]

            # stats has only a few columns identifying the different cases
            # add some more for selecting them
            dfc = cc.cases2df()
            if '[wsp]' in dfc.columns:
                wsp = '[wsp]'
            else:
                wsp = '[Windspeed]'
            # columns we want to add from cc.cases (cases dict) to stats
            cols_cc = set(cols_extra + [wsp])
            # do not add column twice, some might already be in df stats
            add_cols = list(cols_cc - set(df_stats.columns))
            add_cols.append('[case_id]')
            dfc = dfc[add_cols]
            df_stats = pd.merge(df_stats, dfc, on='[case_id]')
            # FIXME: this is very messy, we can end up with both [wsp] and
            # [Windspeed] columns
            if '[Windspeed]' in df_stats.columns and '[wsp]' in df_stats.columns:
                df_stats.drop('[wsp]', axis=1, inplace=True)
            if wsp != '[Windspeed]':
                df_stats.rename(columns={wsp:'[Windspeed]'}, inplace=True)

            # map the run_dir to the same order as the post_dirs, labels
            run_dirs.append(df_stats['[run_dir]'].unique()[0])

            print('%s Cases loaded.' % sim_id)

            # if specified, save the merged sims elsewhere
            if post_dir_save:
                fpath = os.path.join(post_dir_save, '-'.join(sim_ids) + '.h5')
                try:
                    os.makedirs(post_dir_save)
                except OSError:
                    pass
            else:
                fpath = os.path.join(post_dir, '-'.join(sim_ids) + '.h5')
            fmerged = fpath.replace('.h5', '_statistics.h5')
            if ii == 0:
                # and save somewhere so we can add the second data frame on
                # disc
                store = pd.HDFStore(fmerged, mode='w', complevel=9, complib='zlib')
                store.append('table', df_stats, min_itemsize=min_itemsize)
                print(store.get_storer('table').table.description)
                # df_stats.to_hdf(fmerged, 'table', mode='w', format='table',
                #                 complevel=9, complib='blosc')
                print('%s merged stats written to: %s' % (sim_id, fpath))
            else:
                # instead of doing a concat in memory, add to the hdf store
                store.append('table', df_stats)
                # will fail if there are longer string columns compared to ii=0
                # df_stats.to_hdf(fmerged, 'table', mode='r+', format='table',
                #                 complevel=9, complib='blosc', append=True)
                print('%s merging stats into:      %s' % (sim_id, fpath))
#                df_stats = pd.concat([df_stats, df_stats2], ignore_index=True)
#                df_stats2 = None
            # we might run into memory issues
            del df_stats, _, cc
            gc.collect()

        store.close()

        # and load the reduced combined set
        print('loading merged stats:            %s' % fmerged)
        df_stats = pd.read_hdf(fmerged, 'table')
    else:
        sim_id = sim_ids
        sim_ids = [sim_id]
        post_dir = post_dirs
        if isinstance(post_dirs, list):
            post_dir = post_dirs[0]
        cc = sim.Cases(post_dir, sim_id, rem_failed=True)
        df_stats, _, _ = cc.load_stats(columns=columns, leq=False)
        if columns is not None:
            df_stats = df_stats[columns]
        run_dirs = [df_stats['[run_dir]'].unique()[0]]

        # stats has only a few columns identifying the different cases
        # add some more for selecting them
        dfc = cc.cases2df()
        if '[wsp]' in dfc.columns:
            wsp = '[wsp]'
        else:
            wsp = '[Windspeed]'
        # columns we want to add from cc.cases (cases dict) to stats
        cols_cc = set(cols_extra + [wsp])
        # do not add column twice, some might already be in df stats
        add_cols = list(cols_cc - set(df_stats.columns))
        add_cols.append('[case_id]')
        dfc = dfc[add_cols]
        df_stats = pd.merge(df_stats, dfc, on='[case_id]')
        if '[Windspeed]' in df_stats.columns and '[wsp]' in df_stats.columns:
            df_stats.drop('[wsp]', axis=1, inplace=True)
        if wsp != '[Windspeed]':
            df_stats.rename(columns={wsp:'[Windspeed]'}, inplace=True)

    return run_dirs, df_stats

# =============================================================================
### STAT PLOTS
# =============================================================================

def plot_stats2(sim_ids, post_dirs, plot_chans, fig_dir_base=None, labels=None,
                post_dir_save=False, dlc_ignore=['00'], figsize=(8,6),
                eps=False, ylabels=None, title=True, chans_ms_1hz={}):
    """
    Map which channels have to be compared
    """

    # reduce required memory, only use following columns
    cols = ['[run_dir]', '[DLC]', 'channel', '[res_dir]', '[Windspeed]',
            'mean', 'max', 'min', 'std', '[wdir]', '[Case folder]']

    run_dirs, df_stats = merge_sim_ids(sim_ids, post_dirs,
                                       post_dir_save=post_dir_save)

    plot_dlc_stats(df_stats, plot_chans, fig_dir_base, labels=labels,
                   figsize=figsize, dlc_ignore=dlc_ignore, eps=eps,
                   ylabels=ylabels, title=title, chans_ms_1hz=chans_ms_1hz)


def plot_dlc_stats(df_stats, plot_chans, fig_dir_base, labels=None,
                   figsize=(8,6), dlc_ignore=['00'], run_dirs=None,
                   sim_ids=[], eps=False, ylabels=None, title=True,
                   chans_ms_1hz={}):
    """Create for each DLC an overview plot of the statistics.

    df_stats required columns:
    * [DLC]
    * [run_dir]
    * [wdir]
    * [Windspeed]
    * channel
    * stat parameters

    Parameters
    ----------

    df_stats : pandas.DataFrame

    plot_chans : dict
        Dictionary of channels to be plotted. Key is used for the plot title,
        value is a list of unique channel names that will be included for
        the statistic values. For example,
        plot_chans['$B123 M_x$'] = ['blade1-blade1-node-003-momentvec-x',
        'blade2-blade2-node-003-momentvec-x']

    fig_dir_base : str
        Base directory of where to store all the figures. A new sub-directory
        will be created for each DLC.

    labels : list, default=None
        Labels used in the legend when comparing various DLB's

    figsize : tuple, default=(8,6)

    dlc_ignore : list, default=['dlc00']
        By default all but dlc00 (stair case, wind ramp) are plotted. Add
        more dlc numbers here if necessary.

    run_dirs : list, default=None
        If run_dirs is not defined it will be taken from the unique values
        in the DataFrame. The order of the elements in this list needs to be
        consistent with labels, sim_ids (if defined).

    sim_ids : list, default=[]
        Only used when creating the file name of the figures: appended at
        the end of the file name (which starts with the unique channel name).

    chans_ms_1hz : dict, default={}
        Key/value pairs of channel and list of to be plotten m values. Channel
        refers to plot title/label as used as the key value in plot_chans.

    """

    def fig_epilogue(fig, ax, fname_base):
        ax.grid()
        ax.set_xlim(xlims)
        leg = ax.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=3)
        leg.get_frame().set_alpha(0.7)
        title_space = 0.0
        if title:
            fig_title = '%s %s' % (dlc_name, ch_dscr)
            # FIXME: dlc_name is assumed to be not in math mode ($$), so
            # escape underscores to avoid latex going bananas
            if mpl.rcParams['text.usetex']:
                fig_title = '%s %s' % (dlc_name.replace('_', '\\_'), ch_dscr)
            fig.suptitle(fig_title)
            title_space = 0.02
        ax.set_xlabel(xlabel)
        if ylabels is not None:
            ax.set_ylabel(ylabels[ch_name])
        fig.tight_layout()
        spacing = 0.94 - title_space - (0.065 * (ii + 1))
        fig.subplots_adjust(top=spacing)
        fig_path = os.path.join(fig_dir, dlc_name)
        if len(sim_ids)==1:
            fname = fname_base + '.png'
        else:
            fname = '%s_%s.png' % (fname_base, '_'.join(sim_ids))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig_path = os.path.join(fig_path, fname)
        fig.savefig(fig_path)#.encode('latin-1')
        if eps:
            fig.savefig(fig_path.replace('.png', '.eps'))
        fig.clear()
        print('saved: %s' % fig_path)

    mfcs1 = ['k', 'w']
    mfcs2 = ['b', 'w']
    mfcs3 = ['r', 'w']
    mfcs4 = ['k', 'b']
    mark4 = ['s', 'o', '<', '>']
    mfls4 = ['-', '--']

    required = ['[DLC]', '[run_dir]', '[wdir]', '[Windspeed]', '[res_dir]',
                '[Case folder]']
    cols = df_stats.columns
    for col in required:
        if col not in cols:
            print('plot_dlc_stats requires DataFrame with following columns:')
            print(required)
            print('following column is missing in stats DataFrame:', col)
            return

    if run_dirs is None:
        run_dirs = df_stats['[run_dir]'].unique()

    if not sim_ids:
        sim_ids = []
        for run_dir in run_dirs:
            # in case this is a windows path:
            tmp = run_dir.replace('\\', '/').replace(':', '')
            sim_ids.append(tmp.split('/')[-2])

    # first, take each DLC appart
    for gr_name, gr_dlc in df_stats.groupby(df_stats['[Case folder]']):
        dlc_name = gr_name
        if dlc_name[:3].lower() == 'dlc':
            # FIXME: this is messy since this places a hard coded dependency
            # between [Case folder] and [Case id.] when the tag [DLC] is
            # defined in dlcdefs.py
            dlc_name = gr_name.split('_')[0]
        # do not plot the stats for dlc00
        if dlc_name.lower() in dlc_ignore:
            continue
        # cycle through all the target plot channels
        for ch_dscr, ch_names in plot_chans.items():
            # second, group per channel. Note that when the channel names are not
            # identical, we need to manually pick them.
            # figure file name will be the first channel
            if isinstance(ch_names, list):
                ch_name = ch_names[0]
                fname_base = ch_names[0].replace('/', '_')
                df_chan = gr_dlc[gr_dlc.channel.isin(ch_names)]
            else:
                ch_name = ch_names
                ch_names = [ch_names]
                df_chan = gr_dlc[gr_dlc.channel == ch_names]
                fname_base = ch_names.replace('/', '_')

            # if not, than we are missing a channel description, or the channel
            # is simply not available in the given result set
#            if not len(df_chan.channel.unique()) == len(ch_names):
#                continue
            lens = []
            # instead of groupby, select the run_dir in the same order as
            # occuring in the labels and post_dirs lists
            for run_dir in run_dirs:
                lens.append(len(df_chan[df_chan['[run_dir]']==run_dir]))
#            for key, gr_ch_dlc_sid in df_chan.groupby(df_chan['[run_dir]']):
#                lens.append(len(gr_ch_dlc_sid))
            # when the channel is simply not present
            if len(lens) == 0:
                continue
            # when only one of the channels was present, but the set is still
            # complete.
            # FIXME: what if both channels are present?
            if len(ch_names) > 1 and (lens[0] < 1):
                continue
            elif len(ch_names) > 1 and len(lens)==2 and lens[1] < 1:
                continue

            print('start plotting:  %s %s' % (dlc_name.ljust(10), ch_dscr))

            fig, axes = mplutils.make_fig(nrows=1, ncols=1,
                                          figsize=figsize, dpi=120)
            ax = axes[0,0]
            # seperate figure for the mean of the 1Hz equivalent loads
            fig2, axes2 = mplutils.make_fig(nrows=1, ncols=1,
                                            figsize=figsize, dpi=120)
            ax2 = axes2[0,0]

            if fig_dir_base is None and len(sim_ids) < 2:
                res_dir = df_chan['[res_dir]'][:1].values[0]
                fig_dir = os.path.join(fig_dir_base, res_dir)
            elif fig_dir_base is None and len(sim_ids) > 0:
                fig_dir = os.path.join(fig_dir_base, '-'.join(sim_ids))
#            elif fig_dir_base and len(sim_ids) < 2:
#                res_dir = df_chan['[res_dir]'][:1].values[0]
#                fig_dir = os.path.join(fig_dir_base, res_dir)
            elif fig_dir_base is not None:
                # create the compare directory if not defined
                fig_dir = fig_dir_base

            # if we have a list of different cases, we also need to group those
            # because the sim_id wasn't saved before in the data frame,
            # we need to derive that from the run dir
            # if there is only one run dir nothing changes
#            sid_names = []
            # for clarity, set off-set on wind speed when comparing two DLB's
            if len(lens)==2:
                windoffset = [-0.2, 0.2]
                dirroffset = [-5, 5]
            else:
                windoffset = [0]
                dirroffset = [0]
            # in case of a fully empty plot xlims will remain None and there
            # is no need to save the plot
            xlims = None
            # instead of groupby, select the run_dir in the same order as
            # occuring in the labels, post_dirs lists
            for ii, run_dir in enumerate(run_dirs):
                gr_ch_dlc_sid = df_chan[df_chan['[run_dir]']==run_dir]
                if len(gr_ch_dlc_sid) < 1:
                    print('no data for run_dir:', run_dir)
                    continue
#            for run_dir, gr_ch_dlc_sid in df_chan.groupby(df_chan['[run_dir]']):
                if labels is None:
                    sid_name = sim_ids[ii]
                else:
                    sid_name = labels[ii]
#                sid_names.append(sid_name)
                print('   sim_id/label:', sid_name)
                # FIXME: will this go wrong in PY3?
                if dlc_name.lower() in ['dlc61', 'dlc62']:
                    key = '[wdir]'
                    xdata = gr_ch_dlc_sid[key].values + dirroffset[ii]
                    xlabel = 'wind direction [deg]'
                    xlims = [0, 360]
                else:
                    key = '[Windspeed]'
                    xdata = gr_ch_dlc_sid[key].values + windoffset[ii]
                    xlabel = 'Wind speed [m/s]'
                    xlims = [3, 27]
                dmin = gr_ch_dlc_sid['min'].values
                dmean = gr_ch_dlc_sid['mean'].values
                dmax = gr_ch_dlc_sid['max'].values
                dstd = gr_ch_dlc_sid['std'].values
                if len(sim_ids)==1:
                    lab1 = 'mean'
                    lab2 = 'min'
                    lab3 = 'max'
                    lab4 = '1Hz EqL'
                else:
                    lab1 = 'mean %s' % sid_name
                    lab2 = 'min %s' % sid_name
                    lab3 = 'max %s' % sid_name
                    lab4 = '1Hz EqL %s' % sid_name
                mfc1 = mfcs1[ii]
                mfc2 = mfcs2[ii]
                mfc3 = mfcs3[ii]
                ax.errorbar(xdata, dmean, mec='k', marker='o', mfc=mfc1, ls='',
                            label=lab1, alpha=0.7, yerr=dstd, ecolor='k')
                ax.plot(xdata, dmin, mec='b', marker='^', mfc=mfc2, ls='',
                        label=lab2, alpha=0.7)
                ax.plot(xdata, dmax, mec='r', marker='v', mfc=mfc3, ls='',
                        label=lab3, alpha=0.7)

                # mean of 1Hz equivalent loads
                ms = []
                if ch_dscr in chans_ms_1hz:
                    ms = chans_ms_1hz[ch_dscr]
                for im, m in enumerate(ms):
                    # average over seed and possibly yaw angles
                    # wind speed or yaw inflow according to dlc case
                    gr_key = gr_ch_dlc_sid[key]
                    d1hz = gr_ch_dlc_sid[m].groupby(gr_key).mean()

                    ax2.plot(d1hz.index, d1hz.values, mec=mfcs4[ii], alpha=0.7,
                             marker=mark4[im], ls=mfls4[ii], mfc=mfc1,
                             label=lab4, color=mfcs4[ii])

#            for wind, gr_wind in  gr_ch_dlc.groupby(df_stats['[Windspeed]']):
#                wind = gr_wind['[Windspeed]'].values
#                dmin = gr_wind['min'].values#.mean()
#                dmean = gr_wind['mean'].values#.mean()
#                dmax = gr_wind['max'].values#.mean()
##                dstd = gr_wind['std'].mean()
#                ax.plot(wind, dmean, 'ko', label='mean', alpha=0.7)
#                ax.plot(wind, dmin, 'b^', label='min', alpha=0.7)
#                ax.plot(wind, dmax, 'rv', label='max', alpha=0.7)
##                ax.errorbar(wind, dmean, c='k', ls='', marker='s', mfc='w',
##                        label='mean and std', yerr=dstd)
#            if str(dlc_name) not in ['61', '62']:
#                ax.set_xticks(gr_ch_dlc_sid['[Windspeed]'].values)

            # don't save empyt plots
            if xlims is None:
                continue

            fig_epilogue(fig, ax, fname_base)

            # don't save empty plots
            if len(ms) < 1:
                continue

            fig_epilogue(fig2, ax2, fname_base + '_1hz_eql')


class PlotStats(object):

    def __init__(self):
        pass

    def load_stats(self, sim_ids, post_dirs, post_dir_save=False):

        self.sim_ids = sim_ids
        self.post_dirs = post_dirs

        # reduce required memory, only use following columns
        cols = ['[run_dir]', '[DLC]', 'channel', '[res_dir]', '[Windspeed]',
                'mean', 'max', 'min', 'std', '[wdir]']

        # if sim_id is a list, combine the two dataframes into one
        df_stats = pd.DataFrame()
        if type(sim_ids).__name__ == 'list':
            for ii, sim_id in enumerate(sim_ids):
                if isinstance(post_dirs, list):
                    post_dir = post_dirs[ii]
                else:
                    post_dir = post_dirs
                cc = sim.Cases(post_dir, sim_id, rem_failed=True)
                df_stats, _, _ = cc.load_stats(columns=cols, leq=False)
                print('%s Cases loaded.' % sim_id)

                # if specified, save the merged sims elsewhere
                if post_dir_save:
                    fpath = os.path.join(post_dir_save, '-'.join(sim_ids) + '.h5')
                    try:
                        os.makedirs(post_dir_save)
                    except OSError:
                        pass
                else:
                    fpath = os.path.join(post_dir, '-'.join(sim_ids) + '.h5')
                if ii == 0:
                    # and save somewhere so we can add the second data frame on
                    # disc
                    df_stats.to_hdf(fpath, 'table', mode='w', format='table',
                                    complevel=9, complib='blosc')
                    print('%s merged stats written to: %s' % (sim_id, fpath))
                else:
                    # instead of doing a concat in memory, add to the hdf store
                    df_stats.to_hdf(fpath, 'table', mode='r+', format='table',
                                    complevel=9, complib='blosc', append=True)
                    print('%s merging stats into:      %s' % (sim_id, fpath))
                # we might run into memory issues
                del df_stats, _, cc
                gc.collect()
            # and load the reduced combined set
            print('loading merged stats:            %s' % fpath)
            df_stats = pd.read_hdf(fpath, 'table')
        else:
            sim_id = sim_ids
            sim_ids = [sim_id]
            post_dir = post_dirs
            cc = sim.Cases(post_dir, sim_id, rem_failed=True)
            df_stats, _, _ = cc.load_stats(leq=False)

        return df_stats

    def select_extremes_blade_radial(self, df):
        """
        For each radial position of the blade, find the extremes
        """

        def selector(x):
            """
            select following channels:
            'local-blade%1i-node-%03i-momentvec-x'
            'blade2-blade2-node-003-momentvec-y'
            """
            if x[:11] == 'local-blade' and x[22:31] == 'momentvec':
                return True
            else:
                return False

        # find all blade local load channels
        criteria = df.channel.map(lambda x: selector(x))
        df = df[criteria]
        # split channel columns so we can select channels properly
        df = df.join(df.channel.apply(lambda x: pd.Series(x.split('-'))))

        df_ext = {'dlc':[], 'case':[], 'node':[], 'max':[], 'min':[], 'comp':[]}

        def fillvalues(x, ii, maxmin):
            x['node'].append(m_group[3].ix[ii])
            x['dlc'].append(m_group['[DLC]'].ix[ii])
            x['case'].append(m_group['[case_id]'].ix[ii])
            x['comp'].append(m_group[5].ix[ii])
            if maxmin == 'max':
                x['max'].append(m_group['max'].ix[ii])
                x['min'].append(np.nan)
            else:
                x['max'].append(np.nan)
                x['min'].append(m_group['min'].ix[ii])
            return x

        # we take blade1, blade2, and blade3
        df_b2 = df[df[0]=='local']
#        df_b2 = df_b2[df_b2[1]=='blade2']
        df_b2 = df_b2[df_b2[4]=='momentvec']
#        df_b2 = df_b2[df_b2[5]=='x']
        # make sure we only have blade1, 2 and 3
        assert set(df_b2[1].unique()) == set(['blade3', 'blade2', 'blade1'])
#        # number of nodes
#        nrnodes = len(df_b2[3].unique())
        # group by node number, and take the max
        for nodenr, group in df_b2.groupby(df_b2[3]):
            print(nodenr, end='   ')
            for comp, m_group in df_b2.groupby(group[5]):
                print(comp)
                imax = m_group['max'].argmax()
                imin = m_group['min'].argmin()
                df_ext = fillvalues(df_ext, imax, 'max')
                df_ext = fillvalues(df_ext, imin, 'min')

        df_ext = pd.DataFrame(df_ext)
        df_ext.sort(columns='node', inplace=True)

        return df_ext

    def plot_extremes_blade_radial(self, df_ext, fpath):
        nrows = 2
        ncols = 2
        figsize = (11,7.15)
        fig, axes = mplutils.make_fig(nrows=nrows, ncols=ncols, figsize=figsize)

#        self.col = ['r', 'k']
#        self.alf = [1.0, 0.7]
#        self.i = 0

        mx_max = df_ext['max'][df_ext.comp == 'x'].dropna()
        mx_min = df_ext['min'][df_ext.comp == 'x'].dropna()
        my_max = df_ext['max'][df_ext.comp == 'y'].dropna()
        my_min = df_ext['min'][df_ext.comp == 'y'].dropna()
#        nodes = df_ext.node.ix[mx_max.index]
        axes[0,0].plot(mx_max, 'r^', label='$M_{x_{max}}$')
        axes[0,1].plot(mx_min, 'bv', label='$M_{x_{min}}$')
        axes[1,0].plot(my_max, 'r^', label='$M_{y_{max}}$')
        axes[1,1].plot(my_min, 'bv', label='$M_{y_{min}}$')

        axs = axes.ravel()
        for ax in axs:
            ax.grid()
            ax.legend(loc='best')

#        axs[0].set_xticklabels([])
#        axs[1].set_xticklabels([])
#        axs[2].set_xticklabels([])
#        axs[-1].set_xlabel('time [s]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.06)
        fig.subplots_adjust(top=0.98)

        fdir = os.path.dirname(fpath)
#        fname = os.path.basename(fpath)

        if not os.path.exists(fdir):
            os.makedirs(fdir)
        print('saving: %s ...' % fpath, end='')
        fig.savefig(fpath)#.encode('latin-1')
        print('done')
        fig.clear()

        # save as tables
        df_ext.ix[mx_max.index].to_excel(fpath.replace('.png', '_mx_max.xls'))
        df_ext.ix[mx_min.index].to_excel(fpath.replace('.png', '_mx_min.xls'))
        df_ext.ix[my_max.index].to_excel(fpath.replace('.png', '_my_max.xls'))
        df_ext.ix[my_min.index].to_excel(fpath.replace('.png', '_my_min.xls'))

    def extract_leq_blade_radial(self, df_leq, fpath):

        def selector(x):
            """
            select following channels:
            'local-blade%1i-node-%03i-momentvec-x'
            'blade2-blade2-node-003-momentvec-y'
            """
            if x[:11] == 'local-blade' and x[22:31] == 'momentvec':
                return True
            else:
                return False

        # find all blade local load channels
        criteria = df_leq.channel.map(lambda x: selector(x))
        df = df_leq[criteria]
        # split channel columns so we can select channels properly
        df = df.join(df.channel.apply(lambda x: pd.Series(x.split('-'))))
        df.sort(columns=3, inplace=True)
        assert set(df[1].unique()) == set(['blade3', 'blade2', 'blade1'])

        leqs = list(df.keys())[1:10]
        df_ext = {leq:[] for leq in leqs}
        df_ext['node'] = []
        df_ext['comp'] = []

        for nodenr, group in df.groupby(df[3]):
            print(nodenr, end='   ')
            for comp, m_group in df.groupby(group[5]):
                print(comp)
                for leq in leqs:
                    df_ext[leq].append(m_group[leq].max())
                df_ext['node'].append(nodenr)
                df_ext['comp'].append(comp)

        df_ext = pd.DataFrame(df_ext)
        df_ext.sort(columns='node', inplace=True)

        df_ext[df_ext.comp=='x'].to_excel(fpath.replace('.xls', '_x.xls'))
        df_ext[df_ext.comp=='y'].to_excel(fpath.replace('.xls', '_y.xls'))
        df_ext[df_ext.comp=='z'].to_excel(fpath.replace('.xls', '_z.xls'))

        return df_ext


class PlotPerf(object):

    def __init__(self, nrows=4, ncols=1, figsize=(14,11)):

        self.fig, self.axes = mplutils.make_fig(nrows=nrows, ncols=ncols,
                                                 figsize=figsize)
#        self.axs = self.axes.ravel()
        self.col = ['r', 'k']
        self.alf = [1.0, 0.7]
        self.i = 0

    def plot(self, res, label_id):
        """
        """
        i = self.i

        sim_id = label_id
        time = res.sig[:,0]
        self.t0, self.t1 = time[0], time[-1]

        # find the wind speed
        for channame, chan in res.ch_dict.items():
            if channame.startswith('windspeed-global-Vy-0.00-0.00'):
                break
        wind = res.sig[:,chan['chi']]

        chi = res.ch_dict['bearing-shaft_rot-angle_speed-rpm']['chi']
        rpm = res.sig[:,chi]

        chi = res.ch_dict['bearing-pitch1-angle-deg']['chi']
        pitch = res.sig[:,chi]

        chi = res.ch_dict['tower-tower-node-001-momentvec-x']['chi']
        tx = res.sig[:,chi]

        chi = res.ch_dict['tower-tower-node-001-momentvec-y']['chi']
        ty = res.sig[:,chi]

        chi = res.ch_dict['DLL-2-inpvec-2']['chi']
        power = res.sig[:,chi]

        try:
            chi = res.ch_dict['Tors_e-1-100.11']['chi']
        except KeyError:
            chi = res.ch_dict['Tors_e-1-86.50']['chi']
        tors_1 = res.sig[:,chi]

#        try:
#            chi = res.ch_dict['Tors_e-1-96.22']['chi']
#        except:
#            chi = res.ch_dict['Tors_e-1-83.13']['chi']
#        tors_2 = res.sig[:,chi]

        try:
            chi = res.ch_dict['Tors_e-1-84.53']['chi']
        except:
            chi = res.ch_dict['Tors_e-1-73.21']['chi']
        tors_3 = res.sig[:,chi]

        ax = self.axes.ravel()
        ax[0].plot(time, wind, self.col[i]+'--', label='%s wind speed' % sim_id,
                   alpha=self.alf[i])
        ax[0].plot(time, pitch, self.col[i]+'-.', label='%s pitch' % sim_id,
                   alpha=self.alf[i])
        ax[0].plot(time, rpm, self.col[i]+'-', label='%s RPM' % sim_id,
                   alpha=self.alf[i])

        ax[1].plot(time, tx, self.col[i]+'--', label='%s Tower FA' % sim_id,
                   alpha=self.alf[i])
        ax[1].plot(time, ty, self.col[i]+'-', label='%s Tower SS' % sim_id,
                   alpha=self.alf[i])

        ax[2].plot(time, power/1e6, self.col[i]+'-', alpha=self.alf[i],
                   label='%s El Power' % sim_id)

        ax[3].plot(time, tors_1, self.col[i]+'--', label='%s torsion tip' % sim_id,
                   alpha=self.alf[i])
#        ax[3].plot(time, tors_2, self.col[i]+'-.', label='%s torsion 96 pc' % sim_id,
#                   alpha=self.alf[i])
#        ax[3].plot(time, tors_3, self.col[i]+'-', label='%s torsion 84 pc' % sim_id,
#                   alpha=self.alf[i])

        self.i += 1

    def final(self, fig_path, fig_name):

        axs = self.axes.ravel()

        for ax in axs:
            ax.set_xlim([self.t0, self.t1])
            ax.grid()
            ax.legend(loc='best')

        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])
        axs[2].set_xticklabels([])
        axs[-1].set_xlabel('time [s]')

        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.06)
        self.fig.subplots_adjust(top=0.98)

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fname = os.path.join(fig_path, fig_name)
        print('saving: %s ...' % fname, end='')
        self.fig.savefig(fname)#.encode('latin-1')
        print('done')
        self.fig.clear()


def plot_dlc01_powercurve(sim_ids, post_dirs, run_dirs, fig_dir_base):
    """
    Create power curve based on steady DLC01 results
    Use the same format as for HS2 for easy comparison!
    """



def plot_dlc00(sim_ids, post_dirs, run_dirs, fig_dir_base=None, labels=None,
               cnames=['dlc00_stair_wsp04_25_noturb.htc',
                       'dlc00_ramp_wsp04_25_04_noturb.htc'], figsize=(14,11)):
    """
    This version is an update over plot_staircase.
    """

    stairs = []
    # if sim_id is a list, combine the two dataframes into one
    if type(sim_ids).__name__ == 'list':
        for ii, sim_id in enumerate(sim_ids):
            if isinstance(post_dirs, list):
                post_dir = post_dirs[ii]
            else:
                post_dir = post_dirs
            stairs.append(sim.Cases(post_dir, sim_id, rem_failed=True))
    else:
        post_dir = post_dirs
        stairs.append(sim.Cases(post_dir, sim_id, rem_failed=True))

    for cname in cnames:
        fp = PlotPerf(figsize=figsize)
        for i, cc in enumerate(stairs):
            if isinstance(cname, list):
                _cname = cname[i]
            else:
                _cname = cname
            if _cname in cc.cases_fail:
                print('no result for %s' % cc.sim_id)
                continue
            cc.change_results_dir(run_dirs[i])
            try:
                res = cc.load_result_file(cc.cases[_cname])
            except KeyError:
                for k in sorted(cc.cases.keys()):
                    print(k)
                print('-'*79)
                print(cc.sim_id, _cname)
                print('-'*79)
                raise KeyError
            if labels is not None:
                label = labels[i]
            else:
                label = cc.sim_id
            fp.plot(res, label)
        dlcf = 'dlc' + cc.cases[_cname]['[DLC]']
        fig_path = os.path.join(fig_dir_base, dlcf)
        fp.final(fig_path, _cname.replace('.htc', '.png'))


def plot_staircase(sim_ids, post_dirs, run_dirs, fig_dir_base=None,
                   cname='dlc00_stair_wsp04_25_noturb.htc'):
    """
    Default stair and ramp names:

    dlc00_stair_wsp04_25_noturb
    dlc00_ramp_wsp04_25_04_noturb
    """

    stairs = []

    col = ['r', 'k']
    alf = [1.0, 0.7]

    # if sim_id is a list, combine the two dataframes into one
    if type(sim_ids).__name__ == 'list':
        for ii, sim_id in enumerate(sim_ids):
            if isinstance(post_dirs, list):
                post_dir = post_dirs[ii]
            else:
                post_dir = post_dirs
            stairs.append(sim.Cases(post_dir, sim_id, rem_failed=True))
    else:
        sim_id = sim_ids
        sim_ids = [sim_id]
        post_dir = post_dirs
        stairs.append(sim.Cases(post_dir, sim_id, rem_failed=True))

    fig, axes = mplutils.make_fig(nrows=3, ncols=1, figsize=(14,10))
    ax = axes.ravel()

    for i, cc in enumerate(stairs):
        if cname in cc.cases_fail:
            print('no result for %s' % cc.sim_id)
            continue
        cc.change_results_dir(run_dirs[i])
        res = cc.load_result_file(cc.cases[cname])
        respath = cc.cases[cname]['[run_dir]']
        fname = os.path.join(respath, cname)
        df_respost = pd.read_hdf(fname + '_postres.h5', 'table')
        sim_id = cc.sim_id
        time = res.sig[:,0]
        t0, t1 = time[0], time[-1]

        # find the wind speed
        for channame, chan in res.ch_dict.items():
            if channame.startswith('windspeed-global-Vy-0.00-0.00'):
                break
        wind = res.sig[:,chan['chi']]
        chi = res.ch_dict['bearing-pitch1-angle-deg']['chi']
        pitch = res.sig[:,chi]

        chi = res.ch_dict['bearing-shaft_rot-angle_speed-rpm']['chi']
        rpm = res.sig[:,chi]

        chi = res.ch_dict['bearing-pitch1-angle-deg']['chi']
        pitch = res.sig[:,chi]

        chi = res.ch_dict['tower-tower-node-001-momentvec-x']['chi']
        tx = res.sig[:,chi]

        chi = res.ch_dict['tower-tower-node-001-momentvec-y']['chi']
        ty = res.sig[:,chi]

        chi = res.ch_dict['DLL-2-inpvec-2']['chi']
        power = res.sig[:,chi]

        chi = res.ch_dict['DLL-2-inpvec-2']['chi']
        power_mech = df_respost['stats-shaft-power']

        ax[0].plot(time, wind, col[i]+'--', label='%s wind speed' % sim_id,
                   alpha=alf[i])
        ax[0].plot(time, pitch, col[i]+'-.', label='%s pitch' % sim_id,
                   alpha=alf[i])
        ax[0].plot(time, rpm, col[i]+'-', label='%s RPM' % sim_id,
                   alpha=alf[i])

        ax[1].plot(time, tx, col[i]+'--', label='%s Tower FA' % sim_id,
                   alpha=alf[i])
        ax[1].plot(time, ty, col[i]+'-', label='%s Tower SS' % sim_id,
                   alpha=alf[i])

        ax[2].plot(time, power/1e6, col[i]+'-', label='%s El Power' % sim_id,
                   alpha=alf[i])
        ax[2].plot(time, power_mech/1e3, col[i]+'-', alpha=alf[i],
                   label='%s Mech Power' % sim_id)

    ax[0].set_xlim([t0, t1])
    ax[0].grid()
    ax[0].legend(loc='best')
    ax[0].set_xticklabels([])
#    ax[0].set_xlabel('time [s]')

    ax[1].set_xlim([t0, t1])
    ax[1].grid()
    ax[1].legend(loc='best')
    ax[1].set_xticklabels([])
#    ax[1].set_xlabel('time [s]')

    ax[2].set_xlim([t0, t1])
    ax[2].grid()
    ax[2].legend(loc='best')
    ax[2].set_xlabel('time [s]')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.subplots_adjust(top=0.92)

    if not os.path.exists(fig_dir_base):
        os.makedirs(fig_dir_base)
    fig_path = os.path.join(fig_dir_base, '-'.join(sim_ids) + '_stair.png')
    print('saving: %s ...' % fig_path, end='')
    fig.savefig(fig_path)#.encode('latin-1')
    print('done')
    fig.clear()


if __name__ == '__main__':

    # auto configure directories: assume you are running in the root of the
    # relevant HAWC2 model
    # and assume we are in a simulation case of a certain turbine/project
    P_RUN, P_SOURCE, PROJECT, sim_id, P_MASTERFILE, MASTERFILE, POST_DIR \
        = dlcdefs.configure_dirs()

    # -------------------------------------------------------------------------
#    # manually configure all the dirs
#    p_root_remote = '/mnt/hawc2sim'
#    p_root_local = '/home/dave/DTU/Projects/AVATAR/'
#    # project name, sim_id, master file name
#    PROJECT = 'DTU10MW'
#    sim_id = 'C0014'
#    MASTERFILE = 'dtu10mw_master_C0014.htc'
#    # MODEL SOURCES, exchanche file sources
#    P_RUN = os.path.join(p_root_remote, PROJECT, sim_id+'/')
#    P_SOURCE = os.path.join(p_root_local, PROJECT)
#    # location of the master file
#    P_MASTERFILE = os.path.join(p_root_local, PROJECT, 'htc', '_master/')
#    # location of the pre and post processing data
#    POST_DIR = os.path.join(p_root_remote, PROJECT, 'python-prepost-data/')
#    force_dir = P_RUN
    # -------------------------------------------------------------------------

    # PLOT STATS, when comparing cases
    sim_ids = [sim_id]
    run_dirs = [P_RUN]
    figdir = os.path.join(P_RUN, '..', 'dlcplots/%s' % sim_id)

    print('='*79)
    print('   P_RUN: %s' % P_RUN)
    print('P_SOURCE: %s' % P_SOURCE)
    print(' PROJECT: %s' % PROJECT)
    print('  sim_id: %s' % sim_id)
    print('  master: %s' % MASTERFILE)
    print('  figdir: %s' % figdir)
    print('='*79)

    plot_stats2(sim_ids, POST_DIR, fig_dir_base=figdir)
    plot_dlc00(sim_ids, POST_DIR, run_dirs, fig_dir_base=figdir)
