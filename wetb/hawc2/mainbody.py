'''
Created on 01/08/2016

@author: MMPE
'''
from wetb.hawc2.pc_file import PCFile
from wetb.hawc2.ae_file import AEFile
from wetb.hawc2.htc_file import HTCFile
import os
import numpy as np
from wetb.hawc2.st_file import StFile


class MainBody():
    def __init__(self, htcfile, body_name):
        if isinstance(htcfile, str):
            htcfile = HTCFile(htcfile)
        self.htcfile = htcfile
        s = htcfile.new_htc_structure
        main_bodies = {s[k].name[0]:s[k] for k in s.keys() if s[k].name_ == "main_body"}
        self.main_body = main_bodies[body_name]
        self.main_body = s.get_subsection_by_name(body_name)
        self.stFile = StFile(os.path.join(htcfile.modelpath, self.main_body.timoschenko_input.filename[0]))
        #self.c2def = np.array([v.values[1:5] for v in self.main_body.c2_def if v.name_ == "sec"])
        self.concentrated_mass = [cm.values for cm in self.main_body if cm.name_.startswith('concentrated_mass')]
        
    @property
    def c2def(self):
        if not hasattr(self, "_c2def"):
            self._c2def = np.array([v.values[1:5] for v in self.main_body.c2_def if v.name_ == "sec"])
        return self._c2def

    def plot_xz_geometry(self, plt=None):
        if plt is None:
            import matplotlib.pyplot as plt
            plt.figure()
        plt.xlabel("z")
        plt.ylabel("x")
        z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
        plt.plot(self.c2def[:, 2], self.c2def[:, 0],'.-', label='Center line')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) + self.stFile.x_e(z), label='Elastic center')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) + self.stFile.x_cg(z), label='Mass center')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) + self.stFile.x_sh(z), label='Shear center')
        for cm in self.concentrated_mass:
            plt.plot(self.c2def[cm[0]-1,2]+cm[3],self.c2def[cm[0]-1,0]+cm[1],'x', label='Concentrated mass')
        plt.legend()
    
    def plot_yz_geometry(self, plt=None):
        if plt is None:
            import matplotlib.pyplot as plt
            plt.figure()
        plt.xlabel("z")
        plt.ylabel("y")
        z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
        plt.plot(self.c2def[:, 2], self.c2def[:, 1], '.-', label='Center line')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) + self.stFile.y_e(z), label='Elastic center')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) + self.stFile.y_cg(z), label='Mass center')
        plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) + self.stFile.y_sh(z), label='Shear center')
        for cm in self.concentrated_mass:
            plt.plot(self.c2def[cm[0]-1,2]+cm[3],self.c2def[cm[0]-1,1]+cm[2],'x', label='Concentrated mass')
        plt.legend()

        
# 
# 
# class BladeData(object):
#     def plot_xz_geometry(self, plt):
# 
#         z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]), label='Center line')
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) + self.pcFile.chord(z) / 2, label='Leading edge')
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 0]) - self.pcFile.chord(z) / 2, label="Trailing edge")
#         x, y, z = self.hawc2_splines()
#         #plt.plot(z, x, label='Hawc2spline')
# 
#     def plot_yz_geometry(self, plt):
# 
#         z = np.linspace(self.c2def[0, 2], self.c2def[-1, 2], 100)
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]), label='Center line')
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) + self.pcFile.thickness(z) / 100 * self.pcFile.chord(z) / 2, label='Suction side')
#         plt.plot(z, np.interp(z, self.c2def[:, 2], self.c2def[:, 1]) - self.pcFile.thickness(z) / 100 * self.pcFile.chord(z) / 2, label="Pressure side")
#         x, y, z = self.hawc2_splines()
#         #plt.plot(z, y, label='Hawc2spline')
# 
#     def hawc2_splines(self):
#         curve_z = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(self.c2def[:, :3], 1, 0) ** 2, 1)))]
#         curve_z_nd = curve_z / curve_z[-1]
# 
#         def akima(x, y):
#             n = len(x)
#             var = np.zeros((n + 3))
#             z = np.zeros((n))
#             co = np.zeros((n, 4))
#             for i in range(n - 1):
#                 var[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
#             var[n + 1] = 2 * var[n] - var[n - 1]
#             var[n + 2] = 2 * var[n + 1] - var[n]
#             var[1] = 2 * var[2] - var[3]
#             var[0] = 2 * var[1] - var[2]
# 
#             for i in range(n):
#                 wi1 = abs(var[i + 3] - var[i + 2])
#                 wi = abs(var[i + 1] - var[i])
#                 if (wi1 + wi) == 0:
#                     z[i] = (var[i + 2] + var[i + 1]) / 2
#                 else:
#                     z[i] = (wi1 * var[i + 1] + wi * var[i + 2]) / (wi1 + wi)
# 
#             for i in range(n - 1):
#                 dx = x[i + 1] - x[i]
#                 a = (z[i + 1] - z[i]) * dx
#                 b = y[i + 1] - y[i] - z[i] * dx
#                 co[i, 0] = y[i]
#                 co[i, 1] = z[i]
#                 co[i, 2] = (3 * var[i + 2] - 2 * z[i] - z[i + 1]) / dx
#                 co[i, 3] = (z[i] + z[i + 1] - 2 * var[i + 2]) / dx ** 2
#             co[n - 1, 0] = y[n - 1]
#             co[n - 1, 1] = z[n - 1]
#             co[n - 1, 2] = 0
#             co[n - 1, 3] = 0
#             return co
# 
#         def coef2spline(s, co):
#             x, y = [], []
#             for i, c in enumerate(co.tolist()[:-1]):
#                 p = np.poly1d(c[::-1])
#                 z = np.linspace(0, s[i + 1] - s[i ], 10)
#                 x.extend(s[i] + z)
#                 y.extend(p(z))
#             return y
# 
#         x, y, z = [coef2spline(curve_z_nd, akima(curve_z_nd, self.c2def[:, i])) for i in range(3)]
#         return x, y, z



# class Blade(MainBody, BladeData):
#     def __init__(self, htc_filename, modelpath=None, blade_number=1):
#         
#         self.htcfile = htcfile = HTCFile(htc_filename, modelpath)
#         
#         blade_name = [link[2] for link in htcfile.aero if link.name_.startswith('link') and link[0]==blade_number][0]
#         MainBody.__init__(self, htc_filename, modelpath, blade_name)
#         self.pcFile = PCFile(os.path.join(htcfile.modelpath, htcfile.aero.pc_filename[0]),
#                         os.path.join(htcfile.modelpath, htcfile.aero.ae_filename[0]))
#         
#     def plot_xz_geometry(self, plt=None):
#         if plt is None:
#             import matplotlib.pyplot as plt
#             plt.figure()
# 
#         MainBody.plot_xz_geometry(self, plt)
#         BladeData.plot_xz_geometry(self, plt=plt)
#         plt.legend()
#     
#     def plot_geometry_yz(self, plt=None):
#         if plt is None:
#             import matplotlib.pyplot as plt
#             plt.figure()
# 
#         BladeData.plot_yz_geometry(self, plt=plt)
#         MainBody.plot_yz_geometry(self, plt)
#         



