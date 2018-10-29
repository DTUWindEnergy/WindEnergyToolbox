'''
Created on 24/04/2014

@author: MMPE
'''
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from io import open
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()

import numpy as np

class PCFile(object):
    """Read HAWC2 PC (profile coefficients) file

    examples
    --------
    >>> pcfile = PCFile("tests/test_files/NREL_5MW_pc.txt")
    >>> pcfile.CL(21,10) # CL for thickness 21% and AOA=10deg
    1.358
    >>> pcfile.CD(21,10) # CD for thickness 21% and AOA=10deg
    0.0255
    >>> pcfile.CM(21,10) # CM for thickness 21% and AOA=10deg
    -0.1103
    """
    def __init__(self, filename):
        with open (filename) as fid:
            lines = fid.readlines()
        nsets = int(lines[0].split()[0])
        self.pc_sets = {}
        lptr = 1
        for nset in range(1, nsets + 1):
            nprofiles = int(lines[lptr].split()[0])
            lptr += 1
            #assert nprofiles >= 2
            thicknesses = []
            profiles = []
            for profile_nr in range(nprofiles):
                profile_nr, n_rows, thickness = lines[lptr ].split()[:3]
                profile_nr, n_rows, thickness = int(profile_nr), int(n_rows), float(thickness)
                lptr += 1
                data = np.array([[float(v) for v in l.split()[:4]] for l in lines[lptr:lptr + n_rows]])
                thicknesses.append(thickness)
                profiles.append(data)
                lptr += n_rows
            self.pc_sets[nset] = (np.array(thicknesses), profiles)

    def _Cxxx(self, thickness, alpha, column, pc_set_nr=1):
        thicknesses, profiles = self.pc_sets[pc_set_nr]
        index = np.searchsorted(thicknesses, thickness)
        if index == 0:
            index = 1

        Cx0, Cx1 = profiles[index - 1:index + 1]
        Cx0 = np.interp(alpha, Cx0[:, 0], Cx0[:, column])
        Cx1 = np.interp(alpha, Cx1[:, 0], Cx1[:, column])
        th0, th1 = thicknesses[index - 1:index + 1]
        return Cx0 + (Cx1 - Cx0) * (thickness - th0) / (th1 - th0)
    
    def _CxxxH2(self, thickness, alpha, column, pc_set_nr=1):
        thicknesses, profiles = self.pc_sets[pc_set_nr]
        index = np.searchsorted(thicknesses, thickness)
        if index == 0:
            index = 1

        Cx0, Cx1 = profiles[index - 1:index + 1]
        
        Cx0 = np.interp(np.arange(360), Cx0[:,0]+180, Cx0[:,column])
        Cx1 = np.interp(np.arange(360), Cx1[:,0]+180, Cx1[:,column])
        #Cx0 = np.interp(alpha, Cx0[:, 0], Cx0[:, column])
        #Cx1 = np.interp(alpha, Cx1[:, 0], Cx1[:, column])
        th0, th1 = thicknesses[index - 1:index + 1]
        cx = Cx0 + (Cx1 - Cx0) * (thickness - th0) / (th1 - th0)
        return np.interp(alpha+180, np.arange(360), cx)
    
        

    def CL(self, thickness, alpha, pc_set_nr=1):
        """Lift coefficient

        Parameters
        ---------
        thickness : float
            thickness [5]
        alpha : float
            Angle of attack [deg]
        pc_set_nr : int optional
            pc set number, default is 1, normally obtained from ae-file

        Returns
        -------
        Lift coefficient : float
        """
        return self._Cxxx(thickness, alpha, 1, pc_set_nr)


    def CL_H2(self, thickness, alpha, pc_set_nr=1):
        return self._CxxxH2(thickness, alpha, 1, pc_set_nr)
    
    def CD(self, thickness, alpha, pc_set_nr=1):
        """Drag coefficient

        Parameters
        ---------
        radius : float
            radius [m]
        alpha : float
            Angle of attack [deg]
        pc_set_nr : int optional
            pc set number, default is 1, normally obtained from ae-file

        Returns
        -------
        Drag coefficient : float
        """
        return self._Cxxx(thickness, alpha, 2, pc_set_nr)

    def CM(self, thickness, alpha, pc_set_nr=1):
        return self._Cxxx(thickness, alpha, 3, pc_set_nr)

if __name__ == "__main__":
    pcfile = PCFile("tests/test_files/NREL_5MW_pc.txt")
    
    print (pcfile.CL(21,10)) # CL for thickness 21% and AOA=10deg
    #1.358
    print (pcfile.CD(21,10)) # CD for thickness 21% and AOA=10deg
    #0.0255
    print (pcfile.CM(21,10)) # CM for thickness 21% and AOA=10deg
    #-0.1103
