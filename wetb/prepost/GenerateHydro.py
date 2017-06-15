# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:34:15 2016

@author: shfe

Description: This script is used for writing the hydro input files for HAWC2
(the wave type det_airy is not included)
"""

import os

class hydro_input(object):

    """ Basic class aiming for write the hydrodynamics input file"""

    def __init__(self, wavetype, wdepth, spectrum, Hs, Tp, seed, gamma=3.3,
                 stretching=1, wn=None, coef=200, spreading=None,
                 embed_sf=None, embed_sf_t0=None):

        self.wdepth = wdepth
        if self.wdepth < 0:
            raise ValueError('water depth must be greater than 0')

        # Regular Airy Wave Input
        if wavetype == 'reg_airy':
            self.waveno = 0
            self.argument = 'begin %s ;\n\t\tstretching %d;\n\t\twave %.2f %.2f;\n\tend;' \
                            %(wavetype,stretching,Hs,Tp)

        # Iregular Airy Wave Input
        if wavetype == 'ireg_airy':
            self.waveno = 1
            # jonswap spectrum
            if spectrum == 'jonswap':
                spectrumno = 1
                self.argument = 'begin %s ;\n\t\tstretching %d;\n\t\tspectrum %d;'\
                                '\n\t\tjonswap %.2f %.2f %.1f;\n\t\tcoef %d %d;' \
                                %(wavetype,stretching,spectrumno,Hs,Tp,gamma,coef,seed)

            # Pierson Moscowitz spectrum
            elif spectrum == 'pm':
                spectrumno = 2
                self.argument = 'begin %s ;\n\t\tstretching %d;\n\t\tspectrum %d;'\
                                '\n\t\tpm %.2f %.2f ;\n\t\tcoef %d %d;' \
                                %(wavetype,stretching,spectrumno,Hs,
                                  Tp,coef,seed)

            # check the spreading function
            if spreading is not None:
                self.argument += '\n\t\tspreading 1 %d;'%(spreading)
            # check the embeded stream function
            if embed_sf is not None:
                self.argument += '\n\t\tembed_sf %.2f %d;'%(embed_sf, embed_sf_t0)
            self.argument += '\n\tend;'

        # Stream Wave Input
        if wavetype == 'strf':
            self.waveno = 3
            self.argument = 'begin %s ;\n\t\twave %.2f %.2f 0.0;\n\tend;' \
                            %(wavetype,Hs,Tp)

    def execute(self, filename, folder):
        file_path = os.path.join(folder, filename)
        # check if the hydro input file exists
        if os.path.exists(file_path):
            pass
        else:
            # create directory if non existing
            if not os.path.exists(folder):
                os.makedirs(folder)
            FILE = open(file_path,'w+')
            line1 = 'begin wkin_input ;'
            line2 = 'wavetype %d ;' %self.waveno
            line3 = 'wdepth %.1f ;' %self.wdepth
            line4 = 'end ;'
            file_contents = '%s\n\t%s\n\t%s\n\t%s\n%s\n;\nexit;' \
                            %(line1,line2,line3,self.argument,line4)
            FILE.write(file_contents)
            FILE.close()


if __name__ == '__main__':
    hs = 3
    Tp = 11
    hydro = hydro_input(Hs = hs, Tp = Tp, wdepth = 33,spectrum='jonswap',spreading = 2)
    hydro.execute(filename='sss')
