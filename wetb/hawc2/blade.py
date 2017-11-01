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
from wetb.hawc2.ae_file import AEFile
from wetb.hawc2.mainbody import MainBody



# class BladeInfo(object):
#     def twist(self, radius=None):
#         """Aerodynamic twist [deg]. Negative when leading edge is twisted towards wind(opposite to normal definition)\n
# 
#         Parameters
#         ---------
#         radius : float or array_like, optional
#             radius [m] of interest
#             If None (default) the twist of all points are returned
# 
#         Returns
#         -------
#         twist [deg] : float
#         """
#         raise NotImplementedError()
# 
#     def chord(self, radius=None):
#         """Aerodynamic chord
# 
#         Parameters
#         ---------
#         radius : float or array_like, optional
#             radius [m] of interest
#             If None (default) the twist of all points are returned
# 
#         Returns
#         -------
#         chord [m] : float
#         """
#         raise NotImplementedError()
# 
#     
# 
#     def c2nd(self, radius_nd):
#         """Return center line position
# 
#         Parameters
#         ---------
#         radius_nd : float or array_like, optional
#             non dimensional radius
# 
#         Returns
#         -------
#         x,y,z : float
#         """


class H2aeroBlade(PCFile, AEFile, AtTimeFile):
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
    - radius_curved_ac(radius=None) # Curved length of nearest/all aerodynamic calculation points
    
    From ST file
    - radius_st(radius=None, mset=1, set=1)
    - xxx(radius=None, mset=1, set=1) # xxx for each of r, m, x_cg,y_cg, ri_x, ri_y, xs, ys, E, G, Ix, Iy, K, kx, ky, A, pitch, xe, ye
    with template
        
    """

    def __init__(self, htcfile=None, ae_filename=None, pc_filename=None, at_time_filename=None, blade_name=None):
        if htcfile:
            if isinstance(htcfile, str):
                assert htcfile.lower().endswith('.htc')
                htcfile = HTCFile(htcfile)
            self.htcfile = htcfile
            blade_name = blade_name or htcfile.aero.link[2]
            at_time_filename = at_time_filename or ("output_at_time" in htcfile and os.path.join(htcfile.modelpath, htcfile.output_at_time.filename[0] + ".dat"))
            pc_filename = pc_filename or os.path.join(htcfile.modelpath, htcfile.aero.pc_filename[0])
            ae_filename = ae_filename or os.path.join(htcfile.modelpath, htcfile.aero.ae_filename[0])
            self.hawc2_splines_data = self.hawc2_splines()
            
        #mainbodies = [s[k] for k in s.keys() if s[k].name_ == "main_body"]
        #self.mainbody_blade = [mb for mb in mainbodies if mb.name[0] == blade_name][0]
        
        if os.path.isfile(pc_filename) and os.path.isfile(ae_filename):
            AEFile.__init__(self, ae_filename)
            PCFile.__init__(self, pc_filename)
            blade_radius = self.ae_sets[1][-1,0]
        
        if at_time_filename and os.path.isfile(at_time_filename):
            AtTimeFile.__init__(self, at_time_filename, blade_radius)
            self.curved_length = self.radius_curved_ac()[-1]
        else:
            self.curved_length = None
            #z_nd = (np.cos(np.linspace(np.pi, np.pi*2,len(curved_length)-1))+1)/2
            #self.curved_length = np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))[-1]

        
        
        
        
        
    @property
    def c2def(self):
        if not hasattr(self, "_c2def"):
            self._c2def = np.array([v.values[1:5] for v in self.htcfile.blade_c2_def if v.name_ == "sec"])
        return self._c2def
    
        
    def c2nd(self, l_nd, curved_length=True):
        curve_l_nd, x, y, z, twist = self.hawc2_splines_data
        if curved_length:
            l_nd = (np.max([np.min([l_nd, np.ones_like(l_nd)], 0), np.zeros_like(l_nd) + self.c2def[0, 2] / self.c2def[-1, 2]], 0))
            return np.array([np.interp(l_nd, curve_l_nd, xyz) for xyz in [x,y,z, twist]]).T
        else:
            l_nd = (np.max([np.min([l_nd, np.ones_like(l_nd)], 0), np.zeros_like(l_nd)], 0))
            return np.array([np.interp(l_nd, z/z[-1], xyz) for xyz in [x,y,z, twist]]).T
            
    def c2(self, l, curved_length=True):
        if curved_length:
            L = self.curved_length
        else:
            L = self.c2def[-1,2]
        return self.c2nd(l/L, curved_length)
    
    def hawc2_splines(self):
        curve_l = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))]
        curve_l_nd = curve_l / curve_l[-1]

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
            return x,y

        x, y, z, twist = [coef2spline(curve_l_nd, akima(curve_l_nd, self.c2def[:, i]))[1] for i in range(4)]
        curve_l_nd = coef2spline(curve_l_nd, akima(curve_l_nd, self.c2def[:, 0]))[0]
        return curve_l_nd, x, y, z, twist

    def xyztwist(self, l=None, curved_length=False):
        """Return splined x,y,z and twist 
        
        Parameters
        ----------
        l : int, float, arraylike or None, optional
            Position of interest, seee curved_length\n
            If None (default) all x, y, z, and twist defined in c2def
        curved_length : bool, optional
            - If False: l is z coordinate of section
            - If True: l is curved length
            
        Returns
        -------
        x,y,z,twist
                """
        if l is None:
            return self.c2def[:, :3]
        else:
            r_nd = np.linspace(0,1,1000)
            if curved_length:
                #curved_length = np.cumsum(np.sqrt((np.diff(self.c2nd(np.linspace(0,1,100)),1,0)[:,:3]**2).sum(1)))
                return self.c2nd(l/self.radius_curved_ac()[-1])    
#                 z_nd = (np.cos(np.linspace(np.pi, np.pi*2,len(curved_length)-1))+1)/2
#                 assert np.all(l>=curved_length[0]) and np.all(l<=curved_length[-1])
#                 return self.c2nd(r_nd[np.argmin(np.abs(curved_length-l))+1])    
            else:
                assert np.all(l>=self.c2def[0,2]) and np.all(l<=self.c2def[-1,2])
                return self.c2nd(l/self.c2def[-1, 2])
            
    def _Cxxx(self, radius, alpha, column, ae_set_nr=1):
        thickness = self.thickness(radius, ae_set_nr)
        pc_set_nr = self.pc_set_nr(radius, ae_set_nr)
        thicknesses, profiles = self.pc_sets[pc_set_nr]
        index = np.searchsorted(thicknesses, thickness)
        if index == 0:
            index = 1

        Cx0, Cx1 = profiles[index - 1:index + 1]
        Cx0 = np.interp(alpha, Cx0[:, 0], Cx0[:, column])
        Cx1 = np.interp(alpha, Cx1[:, 0], Cx1[:, column])
        th0, th1 = thicknesses[index - 1:index + 1]
        return Cx0 + (Cx1 - Cx0) * (thickness - th0) / (th1 - th0)
    
    def _CxxxH2(self, radius, alpha, column, ae_set_nr=1):
        thickness = self.thickness(radius, ae_set_nr)
        pc_set_nr = self.pc_set_nr(radius, ae_set_nr)
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


    def CL_H2(self, radius, alpha, ae_set_nr=1):
        return self._CxxxH2(radius, alpha, 1, ae_set_nr)
    
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
            
            
    def plot_xz_geometry(self, plt):

        z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]), label='Center line')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) + self.chord(z) / 2, label='Leading edge')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) - self.chord(z) / 2, label="Trailing edge")
        curve_l_nd, x, y, z, twist = self.hawc2_splines()
        plt.plot(z, x, label='Hawc2spline')

    def plot_yz_geometry(self, plt):

        z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]), label='Center line')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) + self.thickness(z) / 100 * self.chord(z) / 2, label='Suction side')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) - self.thickness(z) / 100 * self.chord(z) / 2, label="Pressure side")
        curve_l_nd, x, y, z, twist = self.hawc2_splines()
        plt.plot(z, y, label='Hawc2spline')
            
class H2Blade(H2aeroBlade, MainBody):
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
    - radius_curved_ac(radius=None) # Curved length of nearest/all aerodynamic calculation points
    
    From ST file
    - radius_st(radius=None, mset=1, set=1)
    - xxx(radius=None, mset=1, set=1) # xxx for each of r, m, x_cg,y_cg, ri_x, ri_y, xs, ys, E, G, Ix, Iy, K, kx, ky, A, pitch, xe, ye
    with template
        
    """
    def __init__(self, htcfile=None, ae_filename=None, pc_filename=None, at_time_filename=None, st_filename=None, blade_name=None):
        if htcfile is not None:
            if isinstance(htcfile, str):
                htcfile = HTCFile(htcfile)
            s = htcfile.new_htc_structure
#         at_time_filename = at_time_filename or ("output_at_time" in htcfile and os.path.join(htcfile.modelpath, htcfile.output_at_time.filename[0] + ".dat"))
#         pc_filename = pc_filename or os.path.join(htcfile.modelpath, htcfile.aero.pc_filename[0])
#         ae_filename = ae_filename or os.path.join(htcfile.modelpath, htcfile.aero.ae_filename[0])
#         
            mainbodies = [s[k] for k in s.keys() if s[k].name_ == "main_body"]
            if blade_name is None:
                blade_name = htcfile.aero.link[2]
            self.mainbody_blade = htcfile.new_htc_structure.get_subsection_by_name(blade_name)
            st_filename = st_filename or os.path.join(htcfile.modelpath, self.mainbody_blade.timoschenko_input.filename[0])
            MainBody.__init__(self, htcfile, blade_name)
        elif st_filename and os.path.isfile(st_filename):
            StFile.__init__(self, st_filename)
        H2aeroBlade.__init__(self, htcfile, ae_filename=ae_filename, pc_filename=pc_filename, at_time_filename=at_time_filename, blade_name=blade_name)
        
#     def __init__(self, htcfile, ae_filename=None, pc_filename=None, at_time_filename=None, st_filename=None, blade_name=None):
#         
#         if isinstance(htcfile, str):
#             assert htcfile.lower().endswith('.htc')
#             htcfile = HTCFile(htcfile)
#         
#         blade_name = blade_name or htcfile.aero.link[2]
#         
#         if os.path.isfile(pc_filename) and os.path.isfile(ae_filename):
#             PCFile.__init__(self, pc_filename, ae_filename)
#             blade_radius = self.ae_sets[1][-1,0]
#    
        
#         if os.path.isfile(at_time_filename):
#             AtTimeFile.__init__(self, at_time_filename, blade_radius)
#             self.curved_length = self.radius_curved_ac()[-1]
#         else:
#             raise NotImplementedError
#             #z_nd = (np.cos(np.linspace(np.pi, np.pi*2,len(curved_length)-1))+1)/2
#             #self.curved_length = np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))[-1]
# 
#         self.c2def = np.array([v.values[1:5] for v in mainbody_blade.c2_def if v.name_ == "sec"])
#         
#         self.hawc2_splines_data = self.hawc2_splines()       
    
    @property
    def c2def(self):
        if not hasattr(self, "_c2def"):
            self._c2def = np.array([v.values[1:5] for v in self.mainbody_blade.c2_def if v.name_ == "sec"])
        return self._c2def
   
#         
# class H2aeroBlade(H2Blade):
# 
#     def __init__(self, at_time_filename, ae_filename, pc_filename, htc_filename):
#         """
#         at_time_filename: file name of at time file containing twist and chord data
#         """
#         PCFile.__init__(self, pc_filename, ae_filename)
#         blade_radius = self.ae_sets[1][-1,0]
#         AtTimeFile.__init__(self, at_time_filename, blade_radius)
# 
#         assert('twist' in self.attribute_names)
#         htcfile = HTCFile(htc_filename)
# 
# 
#         self.c2def = np.array([v.values[1:5] for v in htcfile.blade_c2_def if v.name_ == "sec"])
#         #self._c2nd = interp1d(self.c2def[:, 2] / self.c2def[-1, 2], self.c2def[:, :3], axis=0, kind='cubic')
# 
#         ac_radii = self.radius_curved_ac()
#         self.hawc2_splines_data = self.hawc2_splines()
#         self.curved_length = np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))[-1]
#         
# #         c2_axis_length = np.r_[0, np.cumsum(np.sqrt((np.diff(self.c2def[:, :3], 1, 0) ** 2).sum(1)))]
# #         self._c2nd = interp1d(c2_axis_length / c2_axis_length[-1], self.c2def[:, :3], axis=0, kind='cubic')
#         #self._c2nd = interp1d(self.c2def[:,2]/self.c2def[-1,2], self.c2def[:, :3], axis=0, kind='cubic')
# # 
# #     def c2nd(self, r_nd):
# #         r_nd_min = np.zeros_like(r_nd) + self.c2def[0, 2] / self.c2def[-1, 2]
# #         r_nd_max = np.ones_like(r_nd)
# #         r_nd = np.max([np.min([r_nd, r_nd_max], 0), r_nd_min], 0)
# #         return self._c2nd(r_nd)
#     

