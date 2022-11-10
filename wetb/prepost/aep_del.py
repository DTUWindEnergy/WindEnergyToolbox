# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:40:44 2020

@author: dave
"""

import os
from argparse import ArgumentParser
import math

import numpy as np
import pandas as pd

from wetb.dlc.high_level import Weibull_IEC


def aep_del(df_stats, resdir):
    """Works for regular wetb statistics df.
    """

    # note about the unique channel names: they are saved under
    # prepost-data/{sim_id}_unique-channel-names.csv
    # after running parpost.py

    # TODO: make option to have user define list of channels
    # channels = df_stats['channel'].unique()

    # DLC12: assumes that case_id (the file name) has the following format:
    # dlc10_wsp04_wdir000_s1001.htc

    # -----------------------------------------------------
    # USER INPUT
    neq_life = 1e7
    vref = 50 # used for the IEC Weibull distribution
    fact_1year = 365 * 24 * 0.975
    years_del = 20
    # specify which channel to use for the wind speed
    chwind = 'windspeed-global-Vy-0.00-0.00--119.00'
    # specify which channel to use for the electrical power output
    chpowe = 'DLL-generator_servo-inpvec-2'
    yawdist = {'wdir000':0.5, 'wdir010':0.25, 'wdir350':0.25}
    dlc10_dir = 'dlc10_powercurve'
    dlc12_dir = 'dlc12_iec61400-1ed3'
    # -----------------------------------------------------

    df_stats['hour_weight'] = 0

    # -----------------
    # DLC10
    # -----------------
    df_pc_dlc10 = pd.DataFrame()
    resdir_dlc10 = os.path.join(resdir, dlc10_dir)
    df_dlc10 = df_stats[df_stats['res_dir']==resdir_dlc10]
    if len(df_dlc10) > 0:
        df_pivot = df_dlc10.pivot(index='case_id', columns='channel', values='mean')
        df_pivot.sort_values(chwind, inplace=True)
        # wind speed and power
        wsp = df_pivot[chwind].values
        powe = df_pivot[chpowe].values
        # wind speed probability distribution for 1 year
        wsp_dist = Weibull_IEC(vref, wsp) * fact_1year
        # save data into the DataFrame table
        df_pc_dlc10['wsp'] = wsp
        df_pc_dlc10['powe'] = powe
        df_pc_dlc10['hours'] = wsp_dist
        df_pc_dlc10['aep'] = (powe * wsp_dist).sum()

    # -----------------
    # DLC12
    # -----------------
    resdir_dlc12 = os.path.join(resdir, dlc12_dir)
    df_dlc12 = df_stats[df_stats['res_dir']==resdir_dlc12].copy()
    df_pivot = df_dlc12.pivot(index='case_id', columns='channel', values='mean')
    # now add the wsp, yaw, seed columns again
    tmp = pd.DataFrame(df_pivot.index.values)
    df_pivot['dlc'] = ''
    df_pivot['wsp'] = ''
    df_pivot['yaw'] = ''
    df_pivot['seed'] = ''
    # assumes that case_id (the file name) has the following format:
    # dlc12_wsp04_wdir000_s1001.htc
    df_pivot[['dlc', 'wsp', 'yaw', 'seed']] = tmp[0].str.split('_', expand=True).values
    df_pivot['wsp'] = (df_pivot['wsp'].str[3:]).astype(int)

    # sorted on wsp, yaw, seed
    df_pivot.sort_index(inplace=True)

    # figure out how many seeds there are per yaw and wsp
    seeds = {'wsp_yaw':[], 'nr_seeds':[]}
    for ii, grii in df_pivot.groupby('wsp'):
        for jj, grjj in grii.groupby('yaw'):
            seeds['wsp_yaw'].append('%02i_%s' % (ii, jj))
            seeds['nr_seeds'].append(len(grjj))

    nr_seeds = np.array(seeds['nr_seeds']).max()
    seeds_min = np.array(seeds['nr_seeds']).min()
    assert nr_seeds == seeds_min

    # yaw error probabilities
    for yaw, dist in yawdist.items():
        df_pivot.loc[df_pivot['yaw']==yaw, 'hour_weight'] = dist/nr_seeds
    df_pivot['annual_hours_wsp'] = 0

    # Weibull hour distribution as function of wind speeds
    wsps = df_pivot['wsp'].unique()
    wsps.sort()
    wsp_dist = Weibull_IEC(vref, wsps) * fact_1year
    for i, wsp in enumerate(wsps):
        sel = df_pivot['wsp']==wsp
        df_pivot.loc[sel, 'annual_hours_wsp'] = wsp_dist[i]

    df_pivot['annual_hour_dist'] = df_pivot['annual_hours_wsp'] * df_pivot['hour_weight']

    # check we still have the same amount of hours in a year
    hours1 = df_pivot['annual_hour_dist'].sum()
    hours2 = wsp_dist.sum()
    assert np.allclose(hours1, hours2)

    aep = (df_pivot['annual_hour_dist'] * df_pivot[chpowe]).sum()

    df_pc_dlc12 = df_pivot[[chwind, chpowe, 'dlc', 'wsp', 'yaw', 'seed',
                            'hour_weight', 'annual_hours_wsp', 'annual_hour_dist']].copy()
    df_pc_dlc12['aep'] = aep

    # -----------------
    # DLC12 DEL
    ms = [k for k in df_dlc12.columns if k.find('m=') > -1]
    dict_Leq = {m:[] for m in ms}
    dict_Leq['channel'] = []

    # TODO: make option to have user define list of channels
    # for channel in channels:
        #statsel = df_dlc12[df_dlc12['channel']==channel].copy()

    for channel, grch in df_dlc12.groupby('channel'):
        # sort the case_id values in the same way as the pivot table is to
        # align the hours with the corresponding case
        statsel = grch.copy()
        statsel.set_index('case_id', inplace=True)
        statsel = statsel.loc[df_pivot.index,:]

        dict_Leq['channel'].append(channel)

        for m in ms:
            m_ = float(m.split('=')[1])
            eq1hz_mod = np.power(statsel[m].values, m_)
            # R_eq_mod will have to be scaled from its simulation length
            # to 1 hour (hour distribution is in hours...). Since the
            # simulation time has not been multiplied out of R_eq_mod yet,
            # we can just multiply with 3600 (instead of doing 3600/neq)
            tmp = (eq1hz_mod * df_pivot['annual_hour_dist'] * years_del * 3600).sum()
            # the effective Leq for each of the material constants
            leq = math.pow(tmp/neq_life, 1.0/m_)
            dict_Leq[m].append(leq)

    df_leq = pd.DataFrame(dict_Leq)
    df_leq.set_index('channel', inplace=True)

    return df_pc_dlc10, df_pc_dlc12, df_leq


if __name__ == '__main__':

    parser = ArgumentParser(description = "Calculate AEP and LEQ")
    parser.add_argument('--res', type=str, default='res', action='store',
                        dest='resdir', help='Directory containing result files')
    opt = parser.parse_args()

    sim_id = os.path.basename(os.getcwd())
    fname = f'prepost-data/{sim_id}_statistics.h5'
    df_stats = pd.read_hdf(fname, key='table')

    df_pc_dlc10, df_pc_dlc12, df_leq= aep_del(df_stats, opt.resdir)

    # save fot Excel files
    df_leq.to_excel(f'prepost-data/{sim_id}_del.xlsx')
    df_pc_dlc10.to_excel(f'prepost-data/{sim_id}_aep_10.xlsx')
    df_pc_dlc12.to_excel(f'prepost-data/{sim_id}_aep_12.xlsx')
