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

class AEFile(object):

    """Read and write the HAWC2 AE (aerodynamic blade layout) file

    examples
    --------
    >>> aefile = AEFile(r"tests/test_files/NREL_5MW_ae.txt")
    >>> print (aefile.thickness(36)) # Interpolated thickness at radius 36
    23.78048780487805
    >>> print (aefile.chord(36)) # Interpolated chord at radius 36
    3.673
    >>> print (aefile.pc_set_nr(36)) # pc set number at radius 36
    1
    >>> aef = AEFile()
    >>> aef.add_set(11)
    >>> import numpy as np
    >>> aef['1.radius']=np.array([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    >>> aef['1.chord']=np.array([ 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    >>> aef['1.relative_thickness']=np.array([ 100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
    >>> aef['1.pc_set_id']=np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> print(aef.get_data_str())
    1 r[m]           Chord[m]    T/C[%]  Set no.
    1 11
    0.00000000000000000e+00   1.10000000000000009e+00   1.00000000000000000e+02     1
    1.00000000000000006e-01   1.00000000000000000e+00   1.00000000000000000e+02     1
    2.00000000000000011e-01   9.00000000000000022e-01   9.00000000000000000e+01     1
    2.99999999999999989e-01   8.00000000000000044e-01   8.00000000000000000e+01     1
    4.00000000000000022e-01   6.99999999999999956e-01   7.00000000000000000e+01     1
    5.00000000000000000e-01   5.99999999999999978e-01   6.00000000000000000e+01     1
    5.99999999999999978e-01   5.00000000000000000e-01   5.00000000000000000e+01     1
    6.99999999999999956e-01   4.00000000000000022e-01   4.00000000000000000e+01     1
    8.00000000000000044e-01   2.99999999999999989e-01   3.00000000000000000e+01     1
    9.00000000000000022e-01   2.00000000000000011e-01   2.00000000000000000e+01     1
    1.00000000000000000e+00   1.00000000000000006e-01   1.00000000000000000e+01     1
    """

    def __init__(self, filename=None):
        self.ae_sets = {}
        if not filename is None:
            self.read_file(filename)

    def _value(self, radius, column, set_nr=1):
        ae_data = self.ae_sets[set_nr]
        if radius is None:
            return ae_data[:,column]
        else:
            return np.interp(radius, ae_data[:, 0], ae_data[:, column])

    def chord(self, radius=None, set_nr=1):
        return self._value(radius, 1, set_nr)

    def thickness(self, radius=None, set_nr=1):
        return self._value(radius, 2, set_nr)
    
    def radius_ae(self, radius=None, set_nr=1):
        radii = self.ae_sets[set_nr][:,0]
        if radius:
            return radii[np.argmin(np.abs(radii-radius))]
        else:
            return radii
        
    def pc_set_nr(self, radius, set_nr=1):
        ae_data = self.ae_sets[set_nr]
        index = np.searchsorted(ae_data[:, 0], radius)
        index = max(1, index)
        setnrs = ae_data[index - 1:index + 1, 3]
        if setnrs[0] != setnrs[-1]:
            raise NotImplementedError
        return setnrs[0]

    def __getitem__(self, key):
        '''Retrieves the data accoding to keys like "1.chord", with the integer is the set number, and the string is the variable'''

        keys=key.split('.')
        set_nr = 1
        data_key = 'radius'
        if len(keys)==1:
            if len(self.ae_sets)!=1:
                raise KeyError('Must specify the set in the key')
            set_nr = self.ae_sets.keys()[0]
            data_key = keys[0]
        elif len(keys)==2:
            set_nr = int(keys[0])
            data_key = keys[1]
        else:
            raise KeyError('Key member nesting only 2 deep')
        ae_data = self.ae_sets[set_nr]
        if data_key=='radius':
            return ae_data[:,0]
        elif data_key=='chord':
            return ae_data[:,1]
        elif data_key=='relative_thickness':
            return ae_data[:,2]
        elif data_key=='pc_set_id':
            return ae_data[:,3]
        else:
            raise KeyError('That variable does not exist')

    def __setitem__(self, key, val):
        '''Sets the data accoding to keys like "1.chord", with the integer is the set number, and the string is the variable'''

        keys=key.split('.')
        set_nr = 1
        data_key = 'radius'
        if len(keys)==1:
            if len(self.ae_sets)!=1:
                raise KeyError('Must specify the set in the key')
            set_nr = self.ae_sets.keys()[0]
            data_key = keys[0]
        elif len(keys)==2:
            set_nr = int(keys[0])
            data_key = keys[1]
        else:
            raise KeyError('Key member nesting only 2 deep')

        ae_data = self.ae_sets[set_nr]
        if len(val)!=len(ae_data):
            raise IndexError('The assigned data is not consistent with the current data')

        if data_key=='radius':
            ae_data[:,0]=val
        elif data_key=='chord':
            ae_data[:,1]=val
        elif data_key=='relative_thickness':
            ae_data[:,2]=val
        elif data_key=='pc_set_id':
            ae_data[:,3]=val
        else:
            raise KeyError('That variable does not exist')
    
    def keys(self):
        '''Retrieves the keys for this object'''
        retval=[]
        for set_id in self.ae_sets.keys():
            retval.append(str(set_id)+'.'+'radius')
            retval.append(str(set_id)+'.'+'chord')
            retval.append(str(set_id)+'.'+'relative_thickness')
            retval.append(str(set_id)+'.'+'pc_set_id')
        return retval

    def get_set_size(self, set_id):
        '''This will retrieve the number of rows for a given set'''
        if not set_id in self.ae_sets.keys():
            raise KeyError('That set does not exist')
        return len(self.ae_sets[set_id])

    def resize_set(self, set_id, row_nr):
        '''This will change the number of rows for a given set'''
        if not set_id in self.ae_sets.keys():
            raise KeyError('That set does not exist')
        old_data = self.ae_sets[set_id]
        old_rows = len(old_data)
        # early exit if there is nothing to do
        if old_rows==row_nr:
            return
        # must generate a new array
        new_data = np.zeros((row_nr,4))
        asg_rows=old_rows
        if row_nr<asg_rows:
            asg_rows=row_nr
        if asg_rows>0:
            new_data[:asg_rows,:]=old_data[:asg_rows,:]
        self.ae_sets[set_id]=new_data

    def get_set_count(self):
        '''Returns the number of sets'''
        return len(self.ae_sets)

    def add_set(self, row_nr=0):
        '''This method will add another set to the ae data'''
        set_id = 1
        while set_id in self.ae_sets.keys():
            set_id+=1
        self.ae_sets[set_id]=np.zeros((row_nr,4))

    def get_data_str(self):
        '''This method will create a string that is formatted like an ae file with the data in this class'''
        n_sets = len(self.ae_sets)
        retval=str(n_sets)+' r[m]           Chord[m]    T/C[%]  Set no.\n'
        for st_idx, st in self.ae_sets.items():
            retval+=str(st_idx)+' '+str(len(st))+'\n'
            for line in st:
                retval+='%25.17e %25.17e %25.17e %5d\n'%(line[0], line[1], line[2], line[3])
        return retval

    def __str__(self):
        return self.get_data_str()

    def read_file(self, filename):
        ''' This method will read in the ae data from a HAWC2 ae file'''
        with open (filename) as fid:
            lines = fid.readlines()
        nsets = int(lines[0].split()[0])
        lptr = 1
        self.ae_sets = {}
        for _ in range(1, nsets + 1):
            for _ in range(nsets):
                set_nr, n_rows = [int(v) for v in lines[lptr ].split()[:2]]
                lptr += 1
                data = np.array([[float(v) for v in l.split()[:4]] for l in lines[lptr:lptr + n_rows]])
                self.ae_sets[set_nr] = data
                lptr += n_rows






if __name__ == "__main__":
    ae = AEFile(r"tests/test_files/NREL_5MW_ae.txt")
    print (ae.radius_ae(36))
    print (ae.thickness())
    print (ae.chord(36))
    print (ae.pc_set_nr(36))
    aef = AEFile()
    aef.add_set(11)
    import numpy as np
    aef['1.radius']=np.array([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    aef['1.chord']=np.array([ 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    aef['1.relative_thickness']=np.array([ 100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
    aef['1.pc_set_id']=np.array([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    print(aef.get_data_str())

