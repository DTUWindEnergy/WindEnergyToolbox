'''
Created on 04/08/2016

@author: MMPE
'''
from wetb.hawc2.st_file import StFile
import numpy as np


class StSet(object):
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def __call__(self, index):
        return self.data[:, index - 1]

    def __str__(self):
        s = "xxx %d\n" % self.data.shape[0]

        for row in self.data.tolist():
            s += "\t".join(["%-9s" % ("%.4g" % v) for v in row]) + "\n"
        return s


def read_bhawc_stfile(filename, set_nr):
    with open(filename) as fid:
        lines = fid.readlines()
    no_sets = int(lines[0].split()[0])
    start_line = 1
    sets = {}

    for i in range(no_sets):
        while not lines[start_line].startswith(str(i + 1)):
            start_line += 1
        _, n = map(int, lines[start_line].split()[:2])
        sets[i + 1] = StSet([[v for v in row.split()] for row in lines[start_line + 1:start_line + 1 + n]])
        start_line += 1 + n
    return sets[set_nr]


def bhawc2_st19_to_bhawc_st13_converter(st19_filename, set_nr):
    with open(st19_filename) as fid:
        lines = fid.readlines()
    no_sets = int(lines[0].split()[0])
    start_line = 1
    sets = {}

    for i in range(no_sets):
        while not lines[start_line].startswith(str(i + 1)):
            start_line += 1
        _, n = map(int, lines[start_line].split()[:2])
        sets[i + 1] = StSet([[v for v in row.split()] for row in lines[start_line + 1:start_line + 1 + n]])
        start_line += 1 + n
    st = st19 = sets[set_nr]
    st13 = StSet(np.array([st(1),  # 1 Radius [m]
                           st(2),  # 2 mass pr unit length [kg/m]
                           # 3 Polar mass moment of inertia [kgm] (r_iy^2+r_ix^2)*m
                           st(2) * ((st(5) - st(3)) ** 2 + (st(6) - st(4)) ** 2),
                           # 4 x_elastic center [m] converted from PE coordinates to Blade
                           # coordinates (rotate 180 around y)
                           - st(18),
                           st(19),  # 5 y_elastic center [m]
                           st(6),  # 6 structural pitch [deg]
                           st(11) * st(9),  # 7 E*Ix [N m^2]
                           st(12) * st(9),  # 8 E*Iy [N m^2]
                           st(13) * st(10),  # 9 G*Iz [N m^2]
                           st(3),  # 10 x center of gravity  [m]
                           st(4),  # 11 y center of gravity [m]
                           st(7),  # 12 x shear center [m]
                           st(8),  # 13 y shear center [m]
                           ]).T)
    print(st13)


if __name__ == "__main__":
    bhawc2_st19_to_bhawc_st13_converter(
        r'C:\mmpe\HAWC2\models\SWT3.6-107\original_data\B52 data from SWT 2006-09-20/st_B52_K00_DEV00_20160920.st', 1)
