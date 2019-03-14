'''
Created on 22. jun. 2017

@author: mmpe
'''
import os

import numpy as np
from wetb.hawc2.htc_file import HTCFile
from wetb.wind.turbulence import mann_turbulence


class TurbulenceFile(object):
    def __init__(self, filename, Nxyz, dxyz, transport_speed=10, mean_wsp=0, center_position=(0, 0, -20)):
        self.filename = filename
        self.Nxyz = Nxyz
        self.dxyz = dxyz
        self.transport_speed = transport_speed
        self.mean_wsp = mean_wsp
        self.center_position = center_position
        self.data = mann_turbulence.load(filename, Nxyz)

    @property
    def data3d(self):
        return self.data.reshape(self.Nxyz)

    @staticmethod
    def load_from_htc(htcfilename, modelpath=None, type='mann'):
        htc = HTCFile(htcfilename, modelpath)

        Nxyz = np.array([htc.wind[type]['box_dim_%s' % uvw][0] for uvw in 'uvw'])
        dxyz = np.array([htc.wind[type]['box_dim_%s' % uvw][1] for uvw in 'uvw'])
        center_position = htc.wind.center_pos0.values
        wsp = htc.wind.wsp
        return [TurbulenceFile(os.path.join(htc.modelpath, htc.wind[type]['filename_%s' % uvw][0]), Nxyz, dxyz, wsp, (0, wsp)[uvw == 'u'], center_position) for uvw in 'uvw']


if __name__ == '__main__':
    pass
