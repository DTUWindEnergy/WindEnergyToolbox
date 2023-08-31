"""
Created on Wed Oct 10 12:47:10 2018

@author: dave
"""

import os

from wetb.prepost import windIO
from wetb.hawc2 import Hawc2io
from wetb.prepost import hawcstab2

if __name__ == '__main__':

    # =============================================================================
    # READ HAWC2 RESULT FILE
    # =============================================================================

    # METHOD A
    fname = '../wetb/hawc2/tests/test_files/hawc2io/Hawc2ascii.sel'
    res = Hawc2io.ReadHawc2(fname)
    sig = res.ReadAll()

    # METHOD B
    path, file = os.path.dirname(fname), os.path.basename(fname)
    res = windIO.LoadResults(path, file)
    sig = res.sig
    sel = res.ch_details
    # result in dataframe with unique channel names (instead of indices)
    sig_df = res.sig2df()
    ch_df = res.ch_df

    # =============================================================================
    # READ HAWCStab2 files
    # =============================================================================

    res = hawcstab2.results()

    fname = '../wetb/prepost/tests/data/campbell_diagram.cmb'
    df_cmb = res.load_cmb_df(fname)

    fname = '../wetb/prepost/tests/data/dtu10mw_v1_defl_u10000.ind'
    df_ind = res.load_ind(fname)

    fname = '../wetb/prepost/tests/data/dtu10mw.opt'
    df_opt = res.load_operation(fname)

    fname = '../wetb/prepost/tests/data/dtu10mw_v1.pwr'
    df_pwr, units = res.load_pwr_df(fname)

    fname = '../wetb/prepost/tests/data/controller_input_quadratic.txt'
    tuning = hawcstab2.ReadControlTuning()
    tuning.read_parameters(fname)
    # tuning parameters are saved as attributes
    tuning.pi_gen_reg1.K
    tuning.pi_gen_reg2.I
    tuning.pi_gen_reg2.Kp
    tuning.pi_gen_reg2.Ki
    tuning.pi_gen_reg2.Kd
    tuning.pi_pitch_reg3.Kp
    tuning.pi_pitch_reg3.Ki
    tuning.pi_pitch_reg3.Kd
    tuning.pi_pitch_reg3.K1
    tuning.pi_pitch_reg3.K2
    tuning.aero_damp.Kp2
    tuning.aero_damp.Ko1
    tuning.aero_damp.Ko2

    # or you can get them as a dictionary
    tune_tags = tuning.parameters2tags()
