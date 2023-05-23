# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:41:56 2018

@author: dave
"""

import time
import os
from multiprocessing import Pool
from glob import glob
from argparse import ArgumentParser
import json

import numpy as np
import pandas as pd
# import scipy.io as sio
import tables as tbl

from tqdm import tqdm

from wetb.utils.envelope import compute_envelope
from wetb.prepost import windIO
from wetb.prepost.Simulations import EnvelopeClass
from wetb.prepost.simchunks import AppendDataFrames


def logcheck(fname):
    """Check the log file of a single HAWC2 simulation and save results to
    textfile.
    """
    fsave = None
    mode = 'w'

    logf = windIO.LogFile()
    logf.readlog(fname)
    contents = logf._msglistlog2csv('')
    if fsave is None:
        fsave = fname.replace('.log', '.csv')
    with open(fsave, mode) as f:
        f.write(contents)


def add_channels(res):
    """Add channels in memory so they can be included in the statistics calculations
    """

    # EXAMPLE: tower bottom resultant moment
    # chitx = res.ch_dict['tower-tower-node-001-momentvec-x']['chi']
    # chity = res.ch_dict['tower-tower-node-001-momentvec-y']['chi']
    # mx = res.sig[:,chitx]
    # my = res.sig[:,chity]
    # data = np.sqrt(mx*mx + my*my).reshape(len(res.sig),1)
    # # add the channel
    # res.add_channel(data, 'TB_res', 'kNm', 'Tower bottom resultant bending moment')

    # blade resultant loads
    chitx = res.ch_dict['blade1-blade1-node-004-forcevec-x']['chi']
    chity = res.ch_dict['blade1-blade1-node-004-forcevec-y']['chi']
    x = res.sig[:,chitx]
    y = res.sig[:,chity]
    data = np.sqrt(x*x + y*y).reshape(len(res.sig),1)
    res.add_channel(data, 'BR_F_res', 'kN', 'Blade root resultant shear force')

    chitx = res.ch_dict['blade1-blade1-node-004-momentvec-x']['chi']
    chity = res.ch_dict['blade1-blade1-node-004-momentvec-y']['chi']
    x = res.sig[:,chitx]
    y = res.sig[:,chity]
    data = np.sqrt(x*x + y*y).reshape(len(res.sig),1)
    res.add_channel(data, 'BR_M_res', 'kNm', 'Blade root resultant bending moment')

    return res


def loads_at_extreme(dfres, chans_selmax, chans_atmax):
    """For a range of channels defined in chans_list, list their value respective
    values at the time when a max/min occurs.

    It makes most sense to use chans_selmax == chans_atmax (BLADED style reports)

    Other use is to track loads at for example the maximum yaw angle.
    """
    statparams = ['max', 'min']

    # create multi-index: first is the selected channel, second is the stat param
    chans_selmax_ = []
    for k in chans_selmax: chans_selmax_.extend([k ,k])
    index = pd.MultiIndex.from_arrays([chans_selmax_, statparams*len(chans_selmax)])
    # initiale the table
    dfextr = pd.DataFrame(np.zeros((len(chans_selmax_), (len(chans_atmax)))),
                          index=index, columns=chans_atmax)

    # vectorised slightly faster for 8x8 channels
    # # 9.08 ms ± 72.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # select the index at which the max/minimum occurs for the channels in chans_selmax
    idxmax = dfres[chans_selmax].idxmax(axis=0).values
    idxmin = dfres[chans_selmax].idxmin(axis=0).values
    dfextr.loc[(chans_selmax, 'max'), chans_atmax] = dfres.loc[idxmax, chans_atmax].values
    dfextr.loc[(chans_selmax, 'min'), chans_atmax] = dfres.loc[idxmin, chans_atmax].values

    # # looping is slower for 8x8 channels
    # # 11.1 ms ± 21.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # for chan in chans_selmax:
    #     idxmax = dfres[chan].idxmax(axis=0)
    #     idxmin = dfres[chan].idxmin(axis=0)
    #     for atchan in chans_atmax:
    #         dfextr.loc[(chan, 'max'),atchan] = dfres.loc[idxmax, atchan]
    #         dfextr.loc[(chan, 'min'),atchan] = dfres.loc[idxmin, atchan]

    # dfextr.loc[pd.IndexSlice[:,'max'],chan] = dfres.iloc[imax, ichan]
    # dfextr.loc[pd.IndexSlice[:,'min'],chan] = dfres.iloc[imin, ichan]
    # dfextr.loc[pd.IndexSlice[:,:],:].values
    # assert np.allclose(dfextr.loc[:,:].values, dfextr2.loc[:,:].values)

    return dfextr


def envelope(res, int_env=False, Nx=300):

    # define which sensors to use by using the unique_ch_name feature
    ch_list = []

    # find all available outputs for tower, blades of type momentvec Mx
    sel = ((  (res.ch_df['bodyname']=='tower')
            | (res.ch_df['bodyname']=='blade1')
            | (res.ch_df['bodyname']=='blade2')
            | (res.ch_df['bodyname']=='blade3') )
           & (res.ch_df['sensortype']=='momentvec') & (res.ch_df['component']=='x')
          )
    nodeselx = res.ch_df[sel]['unique_ch_name'].values.tolist()

    # all momentvec My
    sel = ((  (res.ch_df['bodyname']=='tower')
            | (res.ch_df['bodyname']=='blade1')
            | (res.ch_df['bodyname']=='blade2')
            | (res.ch_df['bodyname']=='blade3') )
           & (res.ch_df['sensortype']=='momentvec') & (res.ch_df['component']=='y')
          )
    nodesely = set(res.ch_df[sel]['unique_ch_name'].values.tolist())

    for nodex in nodeselx:
        nodey = nodex.replace('-x', '-y')
        # make sure both Mx and My are defined
        if nodey not in nodesely:
            continue
        ch_list.append([nodex, nodey])

    fname = res.FileName + '_envelopes.h5'
    filt = tbl.Filters(complevel=9)
    with tbl.open_file(fname, mode="w", title=str(SIM_ID), filters=filt) as h5f:
        groupname = str(os.path.basename(res.FileName))
        groupname = groupname.replace('-', '_').replace('.', '_')
        ctab = h5f.create_group("/", groupname)

        # envelope = {}
        for ch_names in ch_list:

            ichans = []
            for ch_name in ch_names:
                ichans.append(res.ch_dict[ch_name]['chi'])
            cloud = res.sig[:, ichans]
            # Compute a Convex Hull, the vertices number varies according to
            # the shape of the poligon
            vertices = compute_envelope(cloud, int_env=int_env, Nx=Nx)
            # envelope[ch_names[0]] = vertices

            # # save as simple text file
            # title = ch_names[0]
            # fname = res.FileName.replace('res/', 'prepost-data/env_txt/') + f'_{title}.txt'
            # os.makedirs(os.path.dirname(fname), exist_ok=True)
            # np.savetxt(fname, vertices)

            # save as HDF5 table
            title = str(ch_names[0].replace('-', '_'))
            csv_table = h5f.create_table(ctab, title,
                                          EnvelopeClass.section,
                                          title=title)
            tablerow = csv_table.row
            for row in vertices: #envelope[ch_names[0]]:
                tablerow['Mx'] = float(row[0])
                tablerow['My'] = float(row[1])
                if len(row)>2:
                    tablerow['Mz'] = float(row[2])
                    if len(row)>3:
                        tablerow['Fx'] = float(row[3])
                        tablerow['Fy'] = float(row[4])
                        tablerow['Fz'] = float(row[5])
                    else:
                        tablerow['Fx'] = 0.0
                        tablerow['Fy'] = 0.0
                        tablerow['Fz'] = 0.0
                else:
                    tablerow['Mz'] = 0.0
                    tablerow['Fx'] = 0.0
                    tablerow['Fy'] = 0.0
                    tablerow['Fz'] = 0.0
                tablerow.append()
            csv_table.flush()
    # h5f.close()


def calc(fpath, do_envelope, no_bins, m, atmax):

    t0 = time.time()
    i0 = 0
    i1 = None
    ftype = '.csv'

    # remove the extension
    ext = fpath.split('.')[-1]
    if ext in ('sel', 'dat', 'hdf5'):
        fpath = '.'.join(fpath.split('.')[:-1])
    else:
        print('invalid file extension, ignored:', fpath)
        return

    fdir = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    try:
        res = windIO.LoadResults(fdir, fname, usecols=None, readdata=True)
    except Exception as e:
        print(f'loading {fpath} failed:')
        print(e)
        return
    neq = res.sig[-1,0] - res.sig[0,0]

    # add channels in memory so they can be included in the statistics
    # they are not saved back to disk
    res = add_channels(res)

    # convert to DataFrame
    dfres = res.sig2df()

    # save the envelope
    if do_envelope:
        envelope(res, int_env=False, Nx=300)

    # extremes and corresponding loads at the same time
    if atmax is not None:
        for node, val in atmax.items():
            dfextr = loads_at_extreme(dfres, val['chans_selmax'], val['chans_selmax'])
            dfextr.to_excel(fpath + f'_{node}_loads_at_extreme.xlsx')

    # statistics
    df_stats = res.statsdel_df(i0=i0, i1=i1, statchans='all', neq=neq,
                               no_bins=no_bins, m=m, delchans='all')

    # add path and fname as columns
    df_stats['case_id'] = fname
    df_stats['res_dir'] = fdir

    if ftype == '.csv':
        df_stats.to_csv(fpath+ftype)
    elif ftype == '.h5':
        df_stats.to_hdf(fpath+ftype, 'table', complib='zlib', complevel=9)
    print('% 7.03f  ' % (time.time() - t0), fname)


def par(logdir, resdir, cpus=30, chunksize=30, nostats=False, nolog=False,
        do_envelope=True, no_bins=46, m=[3,4,6,8,9,10,12], atmax=None):
    """Sophia has 32 CPUs per node.
    """

    # log file analysis
    if not nolog:
        files = glob(os.path.join(logdir, '**', '*.log'), recursive=True)
        print(f'start processing {len(files)} logfiles from dir: {logdir}')
        with Pool(processes=cpus) as pool:
            # chunksize = 10 #this may take some guessing ...
            for ind, res in enumerate(pool.imap(logcheck, files), chunksize):
                pass
    print()

    # consider both hdf5 and sel outputs
    if not nostats:
        files = glob(os.path.join(resdir, '**', '*.hdf5'), recursive=True)
        files.extend(glob(os.path.join(resdir, '**', '*.sel'), recursive=True))
        print(f'start processing {len(files)} resfiles from dir: {resdir}')
        # prepare the other arguments
        combis = [(k, do_envelope, no_bins, m, atmax) for k in files]

        # sequential loop
        # for combi in combis:
        #     calc(*combi)

        # start the parallal for-loop using X number of cpus
        with Pool(processes=cpus) as pool:
            # This method chops the iterable into a number of chunks which it submits
            # to the process pool as separate tasks. The (approximate) size of these
            # chunks can be specified by setting chunksize to a positive integer.
            for ind, res in enumerate(pool.starmap(calc, combis), chunksize):
                pass
    print()


def merge_stats(post_dir='prepost-data'):
    """Merge stats for each case into one stats table
    """

    print('start merging statistics into single table...')

    os.makedirs(post_dir, exist_ok=True)
    # -------------------------------------------------------------------------
    # MERGE POSTPRO ON NODE APPROACH INTO ONE DataFrame
    # -------------------------------------------------------------------------
    path_pattern = os.path.join('res', '**', '*.csv')
    csv_fname = '%s_statistics.csv' % SIM_ID
    # if zipchunks:
    #     path_pattern = os.path.join(post_dir, 'statsdel_chnk*.tar.xz')
    fcsv = os.path.join(post_dir, csv_fname)
    mdf = AppendDataFrames(tqdm=tqdm)
    cols = mdf.txt2txt(fcsv, path_pattern, tarmode='r:xz', header=0, sep=',',
                       header_fjoined=None, recursive=True)#, fname_col='[case_id]')
    # and convert to df: takes 2 minutes
    fdf = fcsv.replace('.csv', '.h5')
    store = pd.HDFStore(fdf, mode='w', complevel=9, complib='zlib')
    colnames = cols.split(',')
    # the first column is the channel name but the header is emtpy
    assert colnames[0] == ''
    colnames[0] = 'channel'
    dtypes = {col:np.float64 for col in colnames}
    dtypes['channel'] = str
    dtypes['case_id'] = str
    dtypes['res_dir'] = str
    # when using min_itemsize the column names should be valid variable names
    # mitemsize = {'channel':60, '[case_id]':60}
    mdf.csv2df_chunks(store, fcsv, chunksize=1000000, min_itemsize={}, sep=',',
                      colnames=colnames, dtypes=dtypes, header=0)
    store.close()
    print(f'\nsaved at: {fcsv}')


def merge_logs(post_dir='prepost-data'):

    print('start merging log analysis into single table...')

    os.makedirs(post_dir, exist_ok=True)
    # -------------------------------------------------------------------------
    # MERGE POSTPRO ON NODE APPROACH INTO ONE DataFrame
    # -------------------------------------------------------------------------
    lf = windIO.LogFile()
    path_pattern = os.path.join('logfiles', '**', '*.csv')
    # if zipchunks:
    #     path_pattern = os.path.join(POST_DIR, 'loganalysis_chnk*.tar.xz')
    csv_fname = '%s_ErrorLogs.csv' % SIM_ID
    fcsv = os.path.join(post_dir, csv_fname)
    mdf = AppendDataFrames(tqdm=tqdm)
    # individual log file analysis does not have header, make sure to include
    # a line for the header
    cols = mdf.txt2txt(fcsv, path_pattern, tarmode='r:xz', header=None,
                       header_fjoined=lf._header(), recursive=True)

    # FIXME: this is due to bug in log file analysis. What is going on here??
    # fix that some cases do not have enough columns
    with open(fcsv.replace('.csv', '2.csv'), 'w') as f1:
        with open(fcsv) as f2:
            for line in f2.readlines():
                if len(line.split(';'))==96:
                    line = line.replace(';0.00000000000;nan;-0.0000;',
                                        '0.00000000000;nan;-0.0000;')
                f1.write(line)

    # convert from CSV to DataFrame and save as Excel
    df = lf.csv2df(fcsv.replace('.csv', '2.csv'))
    # since this is mainly text it doesn't make any sense to convert to hdf5
    # for one DLB example, csv is 350 kB, hdf 2 MB.
    # df.to_hdf(fcsv.replace('.csv', '.h5'), 'table')
    df.to_excel(fcsv.replace('.csv', '.xlsx'))
    print(f"\nsaved at: {fcsv.replace('.csv', '2.csv')}")


def save_unique_chan_names(post_dir='prepost-data'):

    fdf = os.path.join(post_dir, '%s_statistics.h5' % SIM_ID)
    df_stats = pd.read_hdf(fdf, 'table')
    chans = df_stats['channel'].unique()
    # TODO: print table with HTC channel names on one side, and the unique
    # channel names on the other. Problem: one HTC line can have multiple channels
    # do not sort, leave in same order as in the sel file
    # chans.sort()
    fname = os.path.join(post_dir, '%s_unique-channel-names.csv' % SIM_ID)
    pd.DataFrame(chans).to_csv(fname)


if __name__ == '__main__':

    parser = ArgumentParser(description = "Parallel post-processing with the "
                            "Python build-in multiprocessing module.")
    parser.add_argument('-n', type=int, default=2, action='store',
                        dest='cpus', help='Number of CPUs to use on the node')
    parser.add_argument('-c', type=int, default=80, action='store',
                        dest='chunksize', help='Chunksize Pool.')

    parser.add_argument('--res', type=str, default='res', action='store',
                        dest='resdir', help='Directory containing result files')
    parser.add_argument('--log', type=str, default='log', action='store',
                        dest='logdir', help='Directory containing HAWC2 log files')

    parser.add_argument('--nologs', default=False, action='store_true',
                        dest='nologs', help='Do not perform the logfile analysis.')
    parser.add_argument('--nostats', default=False, action='store_true',
                        dest='nostats', help='Do not calculate statistics.')
    parser.add_argument('--envelope', default=False, action='store_true',
                        dest='do_envelope', help='Calculate envelopes.')
    parser.add_argument('--no_bins', default=46, action='store',
                        dest='no_bins', help='Number of bins for the rainflow counting.')
    parser.add_argument('--atmax', default=False, action='store', type=str,
                        dest='atmax', help='File name that holds the channels to create '
                        'the table at which maxima and corresponding values are selected.')
    opt = parser.parse_args()

    if opt.atmax:
        with open(opt.atmax) as fio:
            atmax_js = json.load(fio)

    SIM_ID = os.path.basename(os.getcwd())
    par(opt.logdir, opt.resdir, cpus=opt.cpus, chunksize=opt.chunksize,
        nolog=opt.nologs, nostats=opt.nostats, do_envelope=opt.do_envelope,
        no_bins=opt.no_bins, m=[3,4,6,8,9,10,12], atmax=atmax_js)
    if not opt.nostats:
        merge_stats(post_dir='prepost-data')
        save_unique_chan_names(post_dir='prepost-data')
    if not opt.nologs:
        merge_logs(post_dir='prepost-data')
