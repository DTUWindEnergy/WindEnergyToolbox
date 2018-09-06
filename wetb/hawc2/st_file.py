'''
Created on 24/04/2014

@author: MMPE
'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from io import open
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()

import numpy as np

def is_int(v):
    try:
        int(v)
        return True
    except ValueError:
        return False

class StBaseData(object):

    def __init__(self, filename=None):
        self.main_data_sets = {}
        if not filename is None:
            self.read_file(filename)
    
    def _get_column_number(self):
        '''This is suppose to return the number of columns in a data set'''
        raise NotImplementedError('The _get_column_number must be implemented by the derived class')

    def _get_column_labels(self):
        '''This is suppose to be the labels that are added to the st file to help make sense of the data'''
        raise NotImplementedError('The _get_column_labels must be implemented by the derived class')

    def _get_column_keys(self):
        '''These are the keys that identify the columns'''
        raise NotImplementedError('The _get_column_keys must be implemented by the derived class')

    def __getitem__(self, key):
        key_str = self._get_column_keys()
        keys=key.split('.')
        set_id=0
        sub_set_id=0
        data_id=key_str[0]
        sub_set_data = None
        if len(keys)==1:
            if len(self.main_data_sets)==0:
                raise IndexError('No data exists')
            elif len(self.main_data_sets)>1:
                raise IndexError('must give the main-set id in the key when there is more than 1 main set')
            set_id = self.main_data_sets.keys()[0]
            set_data = self.main_data_sets[set_id]
            if len(set_data)==0:
                raise IndexError('No data exists')
            elif len(set_data)>1:
                raise IndexError('Must give the sub-set id when there is more than one sub-set')
            sub_set_id = set_data.keys()[0]
            sub_set_data = set_data[sub_set_id]
            data_id=keys[0]
        elif len(keys)==2:
            if len(self.main_data_sets)==0:
                raise IndexError('No data exists')
            elif len(self.main_data_sets)==1:
                set_id = self.main_data_sets.keys()[0]
                set_data = self.main_data_sets[set_id]
                if len(set_data)==0:
                    raise IndexError('No data exists')
                elif len(set_data)>1:
                    sub_set_id = int(keys[0])
                    if not sub_set_id in set_data.keys():
                        raise IndexError('Sub-set does not exist')
                    sub_set_data = set_data[sub_set_id]
                else:
                    sub_set_id = set_data.keys()[0]
                    sub_set_data = set_data[sub_set_id]
                    test_id = int(keys[0])
                    if test_id!=set_id and test_id!=sub_set_id:
                        raise IndexError('That data does not exist')
            else:
                set_id = int(keys[0])
                if not set_id in self.main_data_sets.keys():
                    raise IndexError('Data set does not exist')
                set_data = self.main_data_sets[set_id]
                if len(set_data)==0:
                    raise IndexError('No data exists')
                elif len(set_data)>1:
                    raise IndexError('Must give the sub-set id when there is more than one sub-set')
                sub_set_id = set_data.keys()[0]
                sub_set_data = set_data[sub_set_id]
            data_id=keys[1]
        elif len(keys)==3:
            set_id = int(keys[0])
            if not set_id in self.main_data_sets.keys():
                raise IndexError('Data set does not exist')
            set_data = self.main_data_sets[set_id]
            sub_set_id = int(keys[1])
            if not sub_set_id in set_data.keys():
                raise IndexError('Sub-set does not exist')
            sub_set_data = set_data[sub_set_id]
            data_id=keys[2]
        else:
            raise IndexError('The keys cannot have more than 3 terms')
        if data_id in key_str:
            col_id = key_str.index(data_id)
            return sub_set_data[:,col_id]
        else:
            raise IndexError('That key does not exist')

    def __setitem__(self, key, val):
        key_str = self._get_column_keys()
        keys=key.split('.')
        set_id=0
        sub_set_id=0
        data_id=key_str[0]
        sub_set_data = None
        if len(keys)==1:
            if len(self.main_data_sets)==0:
                raise IndexError('No data exists')
            elif len(self.main_data_sets)>1:
                raise IndexError('must give the main-set id in the key when there is more than 1 main set')
            set_id = self.main_data_sets.keys()[0]
            set_data = self.main_data_sets[set_id]
            if len(set_data)==0:
                raise IndexError('No data exists')
            elif len(set_data)>1:
                raise IndexError('Must give the sub-set id when there is more than one sub-set')
            sub_set_id = set_data.keys()[0]
            sub_set_data = set_data[sub_set_id]
            data_id=keys[0]
        elif len(keys)==2:
            if len(self.main_data_sets)==0:
                raise IndexError('No data exists')
            elif len(self.main_data_sets)==1:
                set_id = self.main_data_sets.keys()[0]
                set_data = self.main_data_sets[set_id]
                if len(set_data)==0:
                    raise IndexError('No data exists')
                elif len(set_data)>1:
                    sub_set_id = int(keys[0])
                    if not sub_set_id in set_data.keys():
                        raise IndexError('Sub-set does not exist')
                    sub_set_data = set_data[sub_set_id]
                else:
                    sub_set_id = set_data.keys()[0]
                    sub_set_data = set_data[sub_set_id]
                    test_id = int(keys[0])
                    if test_id!=set_id and test_id!=sub_set_id:
                        raise IndexError('That data does not exist')
            else:
                set_id = int(keys[0])
                if not set_id in self.main_data_sets.keys():
                    raise IndexError('Data set does not exist')
                set_data = self.main_data_sets[set_id]
                if len(set_data)==0:
                    raise IndexError('No data exists')
                elif len(set_data)>1:
                    raise IndexError('Must give the sub-set id when there is more than one sub-set')
                sub_set_id = set_data.keys()[0]
                sub_set_data = set_data[sub_set_id]
            data_id=keys[1]
        elif len(keys)==3:
            set_id = int(keys[0])
            if not set_id in self.main_data_sets.keys():
                raise IndexError('Data set does not exist')
            set_data = self.main_data_sets[set_id]
            sub_set_id = int(keys[1])
            if not sub_set_id in set_data.keys():
                raise IndexError('Sub-set does not exist')
            sub_set_data = set_data[sub_set_id]
            data_id=keys[2]
        else:
            raise IndexError('The keys cannot have more than 3 terms')
        if len(sub_set_data)!=len(val):
            raise IndexError('The value is not the same size as the data')
        if data_id in key_str:
            col_id = key_str.index(data_id)
            sub_set_data[:,col_id]=val
        else:
            raise IndexError('That key does not exist')

    def keys(self):
        retval = []
        key_str = self._get_column_keys()
        for set_id, set_data in self.main_data_sets.items():
            for sub_set_id, sub_set_data in set_data.items():
                for key in key_str:
                    retval.append('%d.%d.%s'%(set_id,sub_set_id,key))
        return retval

    def get_data_size(self, set_id, sub_set_id):
        '''Retrieves the size of a specific data set'''
        if not set_id in self.main_data_sets.keys():
            raise IndexError('Data set does not exist')
        set_data = self.main_data_sets[set_id]
        if not sub_set_id in set_data.keys():
            raise IndexError('Sub-set does not exist')
        return len(set_data[sub_set_id])

    def resize_data(self, set_id, sub_set_id, row_nr):
        col_nr = self._get_column_number()
        if not set_id in self.main_data_sets.keys():
            raise IndexError('Data set does not exist')
        set_data = self.main_data_sets[set_id]
        if not sub_set_id in set_data.keys():
            raise IndexError('Sub-set does not exist')
        old_data = set_data[sub_set_id]
        old_rows = len(old_data)
        # do nothing if there is nothing to do
        if old_rows==row_nr:
            return
        new_data = np.zeros((row_nr,col_nr))
        asg_rows = old_rows
        if row_nr<asg_rows:
            asg_rows=row_nr
        new_data[:asg_rows,:]=old_data[:asg_rows,:]
        self.main_data_sets[set_id][sub_set_id]=new_data

    def get_sub_set_count(self, set_id):
        if not set_id in self.main_data_sets.keys():
            raise IndexError('Data set does not exist')
        return len(self.main_data_sets[set_id])

    def add_sub_set(self, set_id, row_nr=0):
        col_nr = self._get_column_number()
        if not set_id in self.main_data_sets.keys():
            raise IndexError('Data set does not exist')
        set_data = self.main_data_sets[set_id]
        sub_set_id=1
        while sub_set_id in set_data.keys():
            sub_set_id+=1
        set_data[sub_set_id]=np.zeros((row_nr,col_nr))

    def get_set_count(self):
        return len(self.main_data_sets)

    def add_set(self, sub_set_cnt=0, row_nr=0):
        col_nr = self._get_column_number()
        set_id = 1
        while set_id in self.main_data_sets.keys():
            set_id+=1
        self.main_data_sets[set_id]={}
        for sub_set_id in range(1,sub_set_cnt+1):
            self.main_data_sets[set_id][sub_set_id]=np.zeros((row_nr,col_nr))

    def get_data_str(self):
        labels = self._get_column_labels()
        retval = str(len(self.main_data_sets.keys()))+'\n'
        for set_id, set_data in self.main_data_sets.items():
            retval+='#'+str(set_id)+'\n'
            for sub_set_id, sub_set_data in set_data.items():
                for line in labels:
                    for term in line:
                        retval+='%25s'%term
                    retval+='\n'
                retval+='$%d %d\n'%(sub_set_id, len(sub_set_data))
                for line in sub_set_data:
                    for term in line:
                        retval+='%25.17e'%term
                    retval+='\n'
        return retval

    def __str__(self):
        return self.get_data_str()

    def read_file(self, filename):

        with open (filename) as fid:
            txt = fid.read()
        main_sets = txt.split("#")
        header_words = main_sets[0].split()
        if len(header_words)>0:
            if not is_int(header_words[0]):
                raise Exception('First term in the header of an ST file must be the number of main sets')
            no_maindata_sets = int(header_words[0])
            assert no_maindata_sets == txt.count("#")
        self.main_data_sets = {}
        for mset in main_sets[1:]:
            mset_nr = int(mset.strip().split()[0])
            set_data_dict = {}

            for set_txt in mset.split("$")[1:]:
                set_lines = set_txt.split("\n")
                set_nr, no_rows = map(int, set_lines[0].split()[:2])
                assert set_nr not in set_data_dict
                set_data_dict[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
            self.main_data_sets[mset_nr] = set_data_dict

class StOrigData(StBaseData):

    _col_labels = [['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'I_p', 'k_x', 'k_y', 'A', 'pitch', 'x_e', 'y_e'],
                ['[m]', '[kg/m]', '[m]', '[m]', '[m]', '[m]', '[m]', '[m]', '[N/m^2]', '[N/m^2]', '[N/m^4]', '[N/m^4]', '[N/m^4]', '[-]', '[-]', '[m^2]', '[deg]', '[m]', '[m]']]

    _col_keys = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'I_p', 'k_x', 'k_y', 'A', 'pitch', 'x_e', 'y_e']

    def __init__(self, filename=None):
        super(StOrigData, self).__init__(filename)

    def _get_column_number(self):
        '''This is suppose to return the number of columns in a data set'''
        return 19

    def _get_column_labels(self):
        '''This is suppose to be the labels that are added to the st file to help make sense of the data'''
        return StOrigData._col_labels
    
    def _get_column_keys(self):
        '''These are the keys that identify the columns'''
        return StOrigData._col_keys

class StFPMData(StBaseData):

    _col_labels = [['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch', 'x_e', 'y_e', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'K22', 'K23', 'K24', 'K25', 'K26', 'K33', 'K34', 'K35', 'K36', 'K44', 'K45', 'K46', 'K55', 'K56', 'K66'],
                ['[m]', '[kg/m]', '[m]', '[m]', '[m]', '[m]', '[deg]', '[m]', '[m]', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2', 'Nm2']]

    _col_keys = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch', 'x_e', 'y_e', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'K22', 'K23', 'K24', 'K25', 'K26', 'K33', 'K34', 'K35', 'K36', 'K44', 'K45', 'K46', 'K55', 'K56', 'K66']

    def __init__(self, filename=None):
        super(StFPMData, self).__init__(filename)

    def _get_column_number(self):
        '''This is suppose to return the number of columns in a data set'''
        return 30

    def _get_column_labels(self):
        '''This is suppose to be the labels that are added to the st file to help make sense of the data'''
        return StFPMData._col_labels

    def _get_column_keys(self):
        '''These are the keys that identify the columns'''
        return StFPMData._col_keys

class StFile(object):
    """Read HAWC2 St (beam element structural data) file

    Methods are autogenerated for:

    - r : curved length distance from main_body node 1 [m]
    - m : mass per unit length [kg/m]
    - x_cg : xc2-coordinate from C1/2 to mass center [m]
    - y_cg : yc2-coordinate from C1/2 to mass center [m]
    - ri_x : radius of gyration related to elastic center. Corresponds to rotation about principal bending xe axis [m]
    - ri_y : radius of gyration related to elastic center. Corresponds to rotation about principal bending ye axis [m]
    - xs : xc2-coordinate from C1/2 to shear center [m]. The shear center is the point where external forces only contributes to pure bending and no torsion.
    - ys : yc2-coordinate from C1/2 to shear center [m]. The shear center is the point where external forces only contributes to pure bending and no torsion.
    - E : modulus of elasticity [N/m2]
    - G : shear modulus of elasticity [N/m2]
    - Ix : area moment of inertia with respect to principal bending xe axis [m4]. This is the principal bending axis most parallel to the xc2 axis
    - Iy : area moment of inertia with respect to principal bending ye axis [m4]
    - K : torsional stiffness constant with respect to ze axis at the shear center [m4/rad]. For a circular section only this is identical to the polar moment of inertia.
    - kx : shear factor for force in principal bending xe direction [-]
    - ky : shear factor for force in principal bending ye direction [-]
    - A : cross sectional area [m2]
    - pitch : structural pitch about z_c2 axis. This is the angle between the xc2 -axis defined with the c2_def command and the main principal bending axis xe.
    - xe : xc2-coordinate from C1/2 to center of elasticity [m]. The elastic center is the point where radial force (in the z-direction) does not contribute to bending around the x or y directions.
    - ye : yc2-coordinate from C1/2 to center of elasticity [m]. The elastic center is the point where radial force (in the

    The autogenerated methods have the following structure

    def xxx(radius=None, mset=1, set=1):
        Parameters:
        -----------
        radius : int, float, array_like or None, optional
            Radius/radii of interest\n
            If int, float or array_like: values are interpolated to requested radius/radii
            If None (default): Values of all radii specified in st file returned
        mset : int, optional
            Main set number
        set : int, optional
            Sub set number


    Examples
    --------
    >>> stfile = StFile(r"tests/test_files/DTU_10MW_RWT_Blade_st.dat")
    >>> print (stfile.m()) # Interpolated mass at radius 36
    [ 1189.51054664  1191.64291781  1202.76694262  ... 15.42438683]
    >>> print (st.E(radius=36, mset=1, set=1))  # Elasticity interpolated to radius 36m
    8722924514.652649
    >>> print (st.E(radius=36, mset=1, set=2))  # Same for stiff blade set
    8.722924514652648e+17
    """
    def __init__(self, filename):
        with open (filename) as fid:
            txt = fid.read()
#         Some files starts with first set ("#1...") with out specifying number of sets
#         no_maindata_sets = int(txt.strip()[0]) 
#         assert no_maindata_sets == txt.count("#")
        self.main_data_sets = {}
        for mset in txt.split("#")[1:]:
            mset_nr = int(mset.strip().split()[0])
            set_data_dict = {}

            for set_txt in mset.split("$")[1:]:
                set_lines = set_txt.split("\n")
                set_nr, no_rows = map(int, set_lines[0].split()[:2])
                assert set_nr not in set_data_dict
                set_data_dict[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
            self.main_data_sets[mset_nr] = set_data_dict

        for i, name in enumerate("r m x_cg y_cg ri_x ri_y x_sh y_sh E G I_x I_y I_p k_x k_y A pitch x_e y_e".split()):
            setattr(self, name, lambda radius=None, mset=1, set=1, column=i: self._value(radius, column, mset, set))

    def _value(self, radius, column, mset_nr=1, set_nr=1):
        st_data = self.main_data_sets[mset_nr][set_nr]
        if radius is None:
            radius = self.radius_st(None, mset_nr, set_nr)
        return np.interp(radius, st_data[:, 0], st_data[:, column])

    def radius_st(self, radius=None, mset=1, set=1):
        r = self.main_data_sets[mset][set][:, 0]
        if radius is None:
            return r
        return r[np.argmin(np.abs(r - radius))]

    def to_str(self, mset=1, set=1):
        d = self.main_data_sets[mset][set]
        return "\n".join([("%12.5e "*d.shape[1]) % tuple(row) for row in d])


if __name__ == "__main__":
    import os
    st = StFile(os.path.dirname(__file__) + r"/tests/test_files/DTU_10MW_RWT_Blade_st.dat")
    print (st.m())
    print (st.E(radius=36, mset=1, set=1))  # Elastic blade
    print (st.E(radius=36, mset=1, set=2))  # stiff blade
    #print (st.radius())
    xyz = np.array([st.x_e(), st.y_e(), st.r()]).T[:40]
    n = 2
    xyz = np.array([st.x_e(None, 1, n), st.y_e(None, 1, n), st.r(None, 1, n)]).T[:40]
    #print (xyz)
    print (np.sqrt(np.sum((xyz[1:] - xyz[:-1]) ** 2, 1)).sum())
    print (xyz[-1, 2])
    print (np.sqrt(np.sum((xyz[1:] - xyz[:-1]) ** 2, 1)).sum() - xyz[-1, 2])
    print (st.x_e(67.8883), st.y_e(67.8883))
    #print (np.sqrt(np.sum(np.diff(xyz, 0) ** 2, 1)))
    print (st.pitch(67.8883 - 0.01687))
    print (st.pitch(23.2446))



    #print (st.)
    #print (st.)
