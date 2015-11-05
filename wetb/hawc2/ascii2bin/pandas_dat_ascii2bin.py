'''
Created on 10/01/2014

@author: MMPE
'''

import numpy as np
from pandas import read_csv



def pandas_dat_ascii2bin(ascii_filename, bin_filename, ui):

    df = ui.exec_long_task("Reading ascii file", False, read_csv, ascii_filename, sep=" ", skipinitialspace=True, header=None)

    def compress(df, bin_filename):
        with open(bin_filename, 'wb') as outfile:
            scale_factors = []
            for _, sensor in df.iteritems():
                sf = sensor.abs().max() / 32000
                if sf > 0:
                    sensor /= sf
                np.round(sensor.values).astype(np.int16).tofile(outfile)
                scale_factors.append(sf)
        return np.array(scale_factors)
    #return compress(df, bin_filename)
    return ui.exec_long_task("Compress and save as binary", False, compress, df, bin_filename)

