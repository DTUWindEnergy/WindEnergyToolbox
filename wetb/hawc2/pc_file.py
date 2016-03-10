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

from wetb.hawc2.ae_file import AEFile
import numpy as np

class PCFile(AEFile):
    """Read HAWC2 PC (profile coefficients) file

    examples
    --------
    >>> pcfile = PCFile("tests/test_files/NREL_5MW_pc.txt", "tests/test_files/NREL_5MW_ae.txt")
    # Same attributes as AEFile
    >>> pcfile.thickness(36) # Interpolated thickness at radius 36
    23.78048780487805
    >>> pcfile.chord(36) # Interpolated chord at radius 36
    3.673
    >>> pcfile.pc_set_nr(36) # pc set number at radius 36
    1
    # Additional attributes
    >>> pcfile.CL(36,10) # CL at radius=36m and AOA=10deg
    1.358
    >>> pcfile.CD(36,10) # CD at radius=36m and AOA=10deg
    0.0255
    >>> pcfile.CM(36,10) # CM at radius=36m and AOA=10deg
    -0.1103
    """
    def __init__(self, filename, ae_filename):
        AEFile.__init__(self, ae_filename)

        with open (filename) as fid:
            lines = fid.readlines()
        nsets = int(lines[0].split()[0])
        self.sets = {}
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
            self.sets[nset] = (np.array(thicknesses), profiles)

    def _Cxxx(self, radius, alpha, column, ae_set_nr=1):
        thickness = self.thickness(radius, ae_set_nr)
        pc_set_nr = self.pc_set_nr(radius, ae_set_nr)
        thicknesses, profiles = self.sets[pc_set_nr]
        index = np.searchsorted(thicknesses, thickness)
        if index == 0:
            index = 1

        Cx0, Cx1 = profiles[index - 1:index + 1]
        Cx0 = np.interp(alpha, Cx0[:, 0], Cx0[:, column])
        Cx1 = np.interp(alpha, Cx1[:, 0], Cx1[:, column])
        th0, th1 = thicknesses[index - 1:index + 1]
        return Cx0 + (Cx1 - Cx0) * (thickness - th0) / (th1 - th0)

    def CL(self, radius, alpha, ae_set_nr=1):
        """Lift coefficient

        Parameters
        ---------
        radius : float
            radius [m]
        alpha : float
            Angle of attack [deg]
        ae_set_nr : int optional
            Aerdynamic set number, default is 1

        Returns
        -------
        Lift coefficient : float
        """
        return self._Cxxx(radius, alpha, 1, ae_set_nr)

    def CD(self, radius, alpha, ae_set_nr=1):
        """Drag coefficient

        Parameters
        ---------
        radius : float
            radius [m]
        alpha : float
            Angle of attack [deg]
        ae_set_nr : int optional
            Aerdynamic set number, default is 1

        Returns
        -------
        Drag coefficient : float
        """
        return self._Cxxx(radius, alpha, 2, ae_set_nr)

    def CM(self, radius, alpha, ae_set_nr=1):

        return self._Cxxx(radius, alpha, 3, ae_set_nr)

if __name__ == "__main__":
    pc = PCFile(r"C:\mmpe\Projects\inflow\Hawc2aero_setup/data/Hawc_pc.b52", r"C:\mmpe\Projects\inflow\Hawc2aero_setup/data/S36_ae_h2.001")
    print (pc)
