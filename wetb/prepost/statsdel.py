# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:13:38 2016

@author: dave
"""
import os

from wetb.prepost import windIO


def logcheck(fname, fsave=None, mode='w'):
    """Check the log file of a single HAWC2 simulation and save results to
    textfile.
    """

    logf = windIO.LogFile()
    logf.readlog(fname)
    contents = logf._msglistlog2csv('')
    if fsave is None:
        fsave = fname.replace('.log', '.csv')
    with open(fsave, mode) as f:
        f.write(contents)


def calc(fpath, no_bins=46, m=[3, 4, 6, 8, 10, 12], neq=None, i0=0, i1=None,
         ftype=False, fsave=False):
    """
    Should we load m, statchans, delchans from another file? This function
    will be called from a PBS script.
    """

    if fpath[-4:] == '.sel' or fpath[-4:] == '.dat':
        fpath = fpath[-4:]

    fdir = os.path.dirname(fpath)
    fname = os.path.basename(fpath)

    res = windIO.LoadResults(fdir, fname, debug=False, usecols=None,
                             readdata=True)
    statsdel = res.statsdel_df(i0=i0, i1=i1, statchans='all', neq=neq,
                               no_bins=no_bins, m=m, delchans='all')

    if fsave:
        if fsave[-4:] == '.csv':
            statsdel.to_csv(fsave)
        elif fsave[-3:] == '.h5':
            statsdel.to_hdf(fsave, 'table', complib='zlib', complevel=9)
    elif ftype == '.csv':
        statsdel.to_csv(fpath+ftype)
    elif ftype == '.h5':
        statsdel.to_hdf(fpath+ftype, 'table', complib='zlib', complevel=9)

    return res, statsdel


if __name__ == '__main__':
    pass
