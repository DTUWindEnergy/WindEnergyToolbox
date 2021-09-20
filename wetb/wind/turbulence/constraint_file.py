import numpy as np
import os
from wetb.hawc2.htc_file import HTCFile


class ConstraintFile(object):

    def __init__(self, center_gl_xyz, box_transport_speed, no_grid_points=(4096, 32, 32), box_size=(6000, 100, 100)):
        """Generate list of constraints for turbulence constraint simulator

        Parameters
        ----------
        center_gl_xyz : (float, float, float)
            Box front plane center position in global coordinates
        box_transport_speed : float or int
            Reference wind speed [m/s] (box transportation speed)
        no_grid_points : [int, int, int]
            Number of grid points in x,y,z direction of the turbulence box
        box_size : [float, float, float]
            Dimension(length) of box in x,y,z direction [m]
        """
        self.center_gl_xyz = center_gl_xyz
        self.box_transport_speed = box_transport_speed
        self.no_grid_points = no_grid_points
        self.box_size = box_size
        self.dxyz = [d / (n - 1) for d, n in zip(self.box_size, self.no_grid_points)]

        self.constraints = {'u': [], 'v': [], 'w': []}

    def load(self, filename):
        """Load constraint from existing constraint file (usable for plotting constraints via time_series()"""
        with open(filename) as fid:
            lines = fid.readlines()
        for l in lines:
            values = l.split(";")
            mx, my, mz = map(int, values[:3])
            comp = values[3]  # {"100":'u','010':'v','001':'w'}["".join(values[3:6])]
            self.constraints[comp].append((mx, my, mz, float(values[-1])))

    def add_constraints(self, glpos, tuvw, subtract_mean=True, fail_outside_box=True, nearest=True):
        """Add constraints to constraint file

        parameters
        ----------
        glpos : (float or array_like, float or array_like, float or array_like)
            global position(s), (x,y,z) of measurement point point(s)\n
            x: horizontal left seen in direction of mean wind, y: direction of mean wind, z: vertical down\n
        tuvw: array_like (shape: no_obs x (1+no_components))
            time and u[, v[, w]] components of wind at measurement point\n
            v and w components are optional\n
        subtract_mean : boolean, optional
            if True, default, the mean values are subtracted from u,v,w
        fail_outside_box : boolean, optional
            if True, default, an error is raised if any positions are outside the box\n
            if False, mann coordinates modulo N is used
        nearest : boolean, optional
            if True, the wsp at the position closest to the mann grid point are applied\n
            if False, the average of all the wsp, that map to the mann grid point are applied
        """
        nx, ny, nz = self.no_grid_points
        dx, dy, dz = self.dxyz
        center_x, center_y, center_z = self.center_gl_xyz

        time, u, v, w = (list(tuvw.T) + [None, None])[:4]
        x, y, z = glpos
        mxs = np.round((time * self.box_transport_speed + (center_y - y)) / dx).astype(int)
        if not np.array(y).shape == mxs.shape:
            y = np.zeros_like(mxs) + y
        if not np.array(x).shape == mxs.shape:
            x = np.zeros_like(mxs) + x
        if not np.array(z).shape == mxs.shape:
            z = np.zeros_like(mxs) + z
        x_err = y - (-(mxs * dx - time * self.box_transport_speed - center_y))
        mys = np.round(((ny - 1) / 2. + (-x + center_x) / dy)).astype(int)
        mzs = np.round(((nz - 1) / 2. + (center_z - z) / dz)).astype(int)

        y_err = x - (-((mys - (ny - 1) / 2) * dy - center_x))
        z_err = z - (-(mzs - (nz - 1) / 2) * dz + center_z)
        pos_err = np.sqrt(x_err**2 + y_err**2 + z_err**2)
        if fail_outside_box:
            for ms, n in zip([mxs, mys, mzs], self.no_grid_points):
                if ms.min() + 1 < 1:
                    i = np.argmin(ms)
                    raise ValueError(
                        "At time, t=%s, global position (%s,%s,%s) maps to grid point (%d,%d,%d) whichs is outside turbulence box, range (1..%d,1..%d,1..%d)" %
                        (time[i], x[i], y[i], z[i], mxs[i] + 1, mys[i] + 1, mzs[i] + 1, nx, ny, nz))
                if ms.max() + 1 > n:
                    i = np.argmax(ms)
                    raise ValueError(
                        "At time, t=%s, global position (%s,%s,%s) maps to grid point (%d,%d,%d) whichs is outside turbulence box, range (1..%d,1..%d,1..%d)" %
                        (time[i], x[i], y[i], z[i], mxs[i] + 1, mys[i] + 1, mzs[i] + 1, nx, ny, nz))
        else:
            mxs %= nx
            mys %= ny
            mzs %= nz
        mann_pos = ["%06d%02d%02d" % (mx, my, mz) for mx, my, mz in zip(mxs, mys, mzs)]
        i = 0
        N = len(mann_pos)
        uvw_lst = [uvw for uvw in [u, v, w] if uvw is not None]
        if subtract_mean:
            uvw_lst = [uvw - uvw.mean() for uvw in uvw_lst]
        while i < N:
            for j in range(i + 1, N + 1):
                if j == N or mann_pos[i] != mann_pos[j]:
                    break
            for comp, wsp in zip(['u', 'v', 'w'], uvw_lst):
                if nearest:
                    value = wsp[i:j][np.argmin(pos_err[i:j])]
                else:  # mean
                    value = wsp[i:j].mean()
                self.constraints[comp].append((mxs[i] + 1, mys[i] + 1, mzs[i] + 1, value))
            i = j

    def __str__(self):
        return "\n".join(["%d;%d;%d;%s;%.10f" % (mx, my, mz, comp, value) for comp in ['u', 'v', 'w']
                          for mx, my, mz, value in self.constraints[comp]])

    def save(self, path, name, folder="./constraints/"):
        path = os.path.join(path, folder)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "%s.con" % name), 'w') as fid:
            fid.write(str(self))

    def time_series(self, y_position=0):
        time = (np.arange(self.no_grid_points[0]) * float(self.dxyz[0]) + y_position) / self.box_transport_speed
        u, v, w = [(np.zeros_like(time) + np.nan)] * 3
        for uvw, constr in zip([u, v, w], [np.array(self.constraints[uvw]) for uvw in 'uvw']):
            if constr.shape[0] > 0:
                uvw[constr[:, 0].astype(int) - 1] = constr[:, 3]
        return time, np.array([u, v, w]).T

    def simulation_cmd(self, ae23, L, Gamma, seed, name, folder="./constraints/"):
        assert isinstance(seed, int) and seed > 0, "seed must be a positive integer"
        cmd = "./csimu2.exe %d %d %d %.3f %.3f %.3f  " % (self.no_grid_points + self.box_size)
        cmd += "%.6f %.2f %.2f %d " % (ae23, L, Gamma, seed)
        cmd += "./turb/%s_s%s %s/%s.con" % (name, seed, folder, name)
        return cmd

    def hawc2_mann_section(self, name, seed):
        htc = HTCFile()
        mann = htc.wind.add_section("mann")
        for uvw in 'uvw':
            setattr(mann, 'filename_%s' % uvw, "./turb/%s_s%04d%s.bin" % (name, int(seed), uvw))
        for uvw, l, n in zip('uvw', self.box_size, self.no_grid_points):
            setattr(mann, 'box_dim_%s' % uvw, [n, l / n])
        mann.dont_scale = 1
        return mann
