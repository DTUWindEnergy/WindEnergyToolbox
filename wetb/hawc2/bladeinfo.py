'''
Created on 3. maj 2017

@author: mmpe
'''
import numpy as np
from wetb.hawc2.at_time_file import AtTimeFile
from wetb.hawc2.pc_file import PCFile
from wetb.hawc2.htc_file import HTCFile
from scipy.interpolate.interpolate import interp1d
import os
from wetb.utils.geometry import rad
from wetb.hawc2.st_file import StFile



class BladeInfo(object):
    def twist(self, radius=None):
        """Aerodynamic twist [deg]. Negative when leading edge is twisted towards wind(opposite to normal definition)\n

        Parameters
        ---------
        radius : float or array_like, optional
            radius [m] of interest
            If None (default) the twist of all points are returned

        Returns
        -------
        twist [deg] : float
        """
        raise NotImplementedError()

    def chord(self, radius=None):
        """Aerodynamic chord

        Parameters
        ---------
        radius : float or array_like, optional
            radius [m] of interest
            If None (default) the twist of all points are returned

        Returns
        -------
        chord [m] : float
        """
        raise NotImplementedError()

    

    def c2nd(self, radius_nd):
        """Return center line position

        Parameters
        ---------
        radius_nd : float or array_like, optional
            non dimensional radius

        Returns
        -------
        x,y,z : float
        """

class H2BladeInfo(BladeInfo, PCFile, AtTimeFile):
    """Provide HAWC2 info about a blade
    
    From AE file:
    - chord(radius=None, set_nr=1):
    - thickness(radius=None, set_nr=1)
    - radius_ae(radius=None, set_nr=1)
    
    From PC file
    - CL(radius, alpha, ae_set_nr=1)
    - CD(radius, alpha, ae_set_nr=1)
    - CM(radius, alpha, ae_set_nr=1)
    
    From at_time_filename
    - attribute_names
    - xxx(radius=None, curved_length=None) # xxx for each attribute name
    - radius_ac(radius=None) # Curved length of nearest/all aerodynamic calculation points
    
    From ST file
    - radius_st(radius=None, mset=1, set=1)
    - xxx(radius=None, mset=1, set=1) # xxx for each of r, m, x_cg,y_cg, ri_x, ri_y, xs, ys, E, G, Ix, Iy, K, kx, ky, A, pitch, xe, ye
    with template
        
    """

    def __init__(self, htcfile, ae_filename=None, pc_filename=None, at_time_filename=None, st_filename=None, blade_name=None):
        
        if isinstance(htcfile, str):
            assert htcfile.lower().endswith('.htc')
            htcfile = HTCFile(htcfile)
        
        blade_name = blade_name or htcfile.aero.link[2]
        s = htcfile.new_htc_structure
        at_time_filename = at_time_filename or os.path.join(htcfile.modelpath, htcfile.output_at_time.filename[0] + ".dat")
        pc_filename = pc_filename or os.path.join(htcfile.modelpath, htcfile.aero.pc_filename[0])
        ae_filename = ae_filename or os.path.join(htcfile.modelpath, htcfile.aero.ae_filename[0])
        
        mainbodies = [s[k] for k in s.keys() if s[k].name_ == "main_body"]
        mainbody_blade = [mb for mb in mainbodies if mb.name[0] == blade_name][0]
        st_filename = st_filename or os.path.join(htcfile.modelpath, mainbody_blade.timoschenko_input.filename[0])
        
        if os.path.isfile(pc_filename) and os.path.isfile(ae_filename):
            PCFile.__init__(self, pc_filename, ae_filename)
            blade_radius = self.ae_sets[1][-1,0]
        
        if os.path.isfile(st_filename):
            StFile.__init__(self, st_filename)
        if os.path.isfile(at_time_filename):
            AtTimeFile.__init__(self, at_time_filename, blade_radius)
        

        self.c2def = np.array([v.values[1:5] for v in mainbody_blade.c2_def if v.name_ == "sec"])

        #self.c2nd = lambda x : interp1d(self.c2def[:, 2] / self.c2def[-1, 2], self.c2def[:, :3], axis=0, kind='cubic')(np.max([np.min([x, np.ones_like(x)], 0), np.zeros_like(x) + self.c2def[0, 2] / self.c2def[-1, 2]], 0))
        x, y, z, twist = self.hawc2_splines()
        def interpolate(r):
            r = (np.max([np.min([r, np.ones_like(r)], 0), np.zeros_like(r) + self.c2def[0, 2] / self.c2def[-1, 2]], 0))
            return np.array([np.interp(r, np.array(z) / z[-1], xyz) for xyz in [x,y,z, twist]]).T
            
        self.c2nd = interpolate #lambda r : interp1d(np.array(z) / z[-1], np.array([x, y, z]).T, axis=0, kind=1)(np.max([np.min([r, np.ones_like(r)], 0), np.zeros_like(r) + self.c2def[0, 2] / self.c2def[-1, 2]], 0))
        
    def hawc2_splines(self):
        curve_z = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))]
        curve_z_nd = curve_z / curve_z[-1]

        def akima(x, y):
            n = len(x)
            var = np.zeros((n + 3))
            z = np.zeros((n))
            co = np.zeros((n, 4))
            for i in range(n - 1):
                var[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            var[n + 1] = 2 * var[n] - var[n - 1]
            var[n + 2] = 2 * var[n + 1] - var[n]
            var[1] = 2 * var[2] - var[3]
            var[0] = 2 * var[1] - var[2]

            for i in range(n):
                wi1 = abs(var[i + 3] - var[i + 2])
                wi = abs(var[i + 1] - var[i])
                if (wi1 + wi) == 0:
                    z[i] = (var[i + 2] + var[i + 1]) / 2
                else:
                    z[i] = (wi1 * var[i + 1] + wi * var[i + 2]) / (wi1 + wi)

            for i in range(n - 1):
                dx = x[i + 1] - x[i]
                a = (z[i + 1] - z[i]) * dx
                b = y[i + 1] - y[i] - z[i] * dx
                co[i, 0] = y[i]
                co[i, 1] = z[i]
                co[i, 2] = (3 * var[i + 2] - 2 * z[i] - z[i + 1]) / dx
                co[i, 3] = (z[i] + z[i + 1] - 2 * var[i + 2]) / dx ** 2
            co[n - 1, 0] = y[n - 1]
            co[n - 1, 1] = z[n - 1]
            co[n - 1, 2] = 0
            co[n - 1, 3] = 0
            return co

        def coef2spline(s, co):
            x, y = [], []
            for i, c in enumerate(co.tolist()[:-1]):
                p = np.poly1d(c[::-1])
                z = np.linspace(0, s[i + 1] - s[i ], 10, endpoint=i >= co.shape[0] - 2)
                x.extend(s[i] + z)
                y.extend(p(z))
            return y

        x, y, z, twist = [coef2spline(curve_z_nd, akima(curve_z_nd, self.c2def[:, i])) for i in range(4)]
        return x, y, z, twist

    def c2def_twist(self, radius=None):
        if radius is None:
            return self.c2def[:, 3]
        else:
            return np.interp(radius, self.c2def[:, 2], self.c2def[:, 3])



class H2aeroBladeInfo(H2BladeInfo):

    def __init__(self, at_time_filename, ae_filename, pc_filename, htc_filename):
        """
        at_time_filename: file name of at time file containing twist and chord data
        """
        PCFile.__init__(self, pc_filename, ae_filename)
        blade_radius = self.ae_sets[1][-1,0]
        AtTimeFile.__init__(self, at_time_filename, blade_radius)

        assert('twist' in self.attribute_names)
        htcfile = HTCFile(htc_filename)


        self.c2def = np.array([v.values[1:5] for v in htcfile.blade_c2_def if v.name_ == "sec"])
        #self._c2nd = interp1d(self.c2def[:, 2] / self.c2def[-1, 2], self.c2def[:, :3], axis=0, kind='cubic')

        ac_radii = self.ac_radius()
        c2_axis_length = np.r_[0, np.cumsum(np.sqrt((np.diff(self.c2def[:, :3], 1, 0) ** 2).sum(1)))]
        self._c2nd = interp1d(c2_axis_length / c2_axis_length[-1], self.c2def[:, :3], axis=0, kind='cubic')
        #self._c2nd = interp1d(self.c2def[:,2]/self.c2def[-1,2], self.c2def[:, :3], axis=0, kind='cubic')

    def c2nd(self, r_nd):
        r_nd_min = np.zeros_like(r_nd) + self.c2def[0, 2] / self.c2def[-1, 2]
        r_nd_max = np.ones_like(r_nd)
        r_nd = np.max([np.min([r_nd, r_nd_max], 0), r_nd_min], 0)
        return self._c2nd(r_nd)
    

