'''
Created on 23/09/2014

@author: MMPE
'''

import numpy as np
import glob
import os
HOURS_PR_YEAR = 365.0 * 24.0


def Weibull(u, k, start, stop, step):
    C = 2 * u / np.sqrt(np.pi)
    cdf = lambda x :-np.exp(-(x / C) ** k)
    wsprange = (np.arange(start, stop + step * 0.01, step)).tolist()
    return {wsp:-cdf(wsp - step / 2) + cdf(wsp + step / 2) for wsp in wsprange}


def dlc_dict(Vin=4, Vr=12, Vout=26, Vref=50, Vstep=2, shape_k=2):
    weibull = Weibull(Vref * 0.2, shape_k, Vin, Vout, Vstep)

    return {  #dlc : (hour_dist, yaw, wsp
            '12': (.975, weibull, {0:.5, 10:.25, -10:.25}),  #normal
            '64': (.025, Weibull(Vref * 0.2, shape_k, Vin, Vref * 0.7, Vstep), {8:.5, -8:.5}),  #idle
            '24': (50.0 / HOURS_PR_YEAR, weibull, {-20:.5, 20:.5}),  # Yaw error
            '31': (1100 * 100 / 3600 / HOURS_PR_YEAR, {Vin:1000 / 1100, Vr:50 / 1100, Vout:50 / 1100}, {0:1}),  #start ups
            '41': (1100 * 100 / 3600 / HOURS_PR_YEAR, {Vin:1000 / 1100, Vr:50 / 1100, Vout:50 / 1100}, {0:1})  #shut down
            }


def file_hour_lst(path, dlc_dict, dlc_folder="DLC%s_IEC61400-1ed3/",
                  dlc_name="dlc%s_wsp%02d_wdir%03d_s*.sel", years=20.0):
    """Create a list of (filename, hours_pr_year) that can be used as input for LifeTimeEqLoad

    Parameters
    ----------
    path : str
        path to result folder, i.e. dlc12 result files are found in path + "DLC12_IEC61400-1ed3/*.sel"
    dlc_dict : dict
        Dictionary of {dlc_id: (dlc_prob, wsp_dict, wdir_dict, wsp),...} where\n
        - dlc_id is the design load case id, e.g. '12'\n
        - dlc_prob is the probability of the dlc_id, e.g. 0.95 for dlc12, i.e 95% normal operation\n
        - wsp_dict is a dictionary of {wsp: wsp_prob,...} see dlc_fatigue.Weibull where\n
            - wsp is a wind speed, e.g. 10 for 10m/s
            - wsp_prob is the probability of the wind speed, e.g. .14 for 14%
        - wdir_dict is a dictionary of {wdir: wdir_prob,...} where\n
            - wdir is a wind direction(yaw error), e.g. 10 for 10 deg yaw error
            - wdir_prob is the probability of the wind direction(yaw error), e.g. .25 for 25%
    dlc_folder : str, default="DLC%s_IEC61400-1ed3/"
        String with the DLC subfolder names. One string substitution is required
        (%s), and should represent the DLC number (withouth comma or point)
    dlc_name : str, default="dlc%s_wsp%02d_wdir%03d_s*.sel"
        String with the DLC names. One string, and two integer substitutions
        are required (%s, %02d, %03d), indicating the DLC number (e.g. '12'),
        the windspeed (e.g. int(6)), and wind speed direction (e.g. int(10))
        respectively. Notice that different seed numbers are covered with the
        wildcard *.
    years : float, default=20.0
        Life time years.
    Returns
    -------
    file_hour_lst : list
        [(filename, hours),...] where\n
        - filename is the name of the file, including path
        - hours is the number of hours pr. 20 year (or whatever is defined in
        the `years` variable) of this file
    """

    fh_lst = []
    for dlc_id in sorted(dlc_dict.keys()):
        dlc_dist, wsp_dict, wdir_dict = dlc_dict[dlc_id]
        for wsp in sorted(wsp_dict.keys()):
            wsp_dist = wsp_dict[wsp]
            for wdir in sorted(wdir_dict.keys()):
                wdir_dist = wdir_dict[wdir]
                folder = os.path.join(path, dlc_folder % dlc_id)
                name = dlc_name % (dlc_id, wsp, wdir % 360)
                files = glob.glob(os.path.join(folder, name))
                for f in sorted(files):
                    f_prob = dlc_dist * wsp_dist * wdir_dist / len(files)
                    f_hours_lifetime = years * HOURS_PR_YEAR * f_prob
                    fh_lst.append((f, f_hours_lifetime))
    return fh_lst
