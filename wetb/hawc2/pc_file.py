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

class PCData(object):

    def __init__(self, filename=None):
        self.sets = {}
        if not filename is None:
            self.read_file(filename)

    def __getitem__(self, key):

        '''Retrieve a value according to keys like '1.0.alpha' where the first is the data, the second is the profile and the third is the variable'''
        keys=key.split('.')
        set_nr = 1
        pfl_nr = 1
        data_key = 'alpha'
        if len(keys)==2:
            if len(self.sets)!=1:
                raise KeyError('Must specify the set in the key')
            set_nr = self.sets.keys()[0]
            pfl_nr = int(keys[0])
            data_key = keys[1]
        elif len(keys)==3:
            set_nr = int(keys[0])
            pfl_nr = int(keys[1])
            data_key = keys[2]
        else:
            raise KeyError('Key member nesting only 2 deep')
        thickness = self.sets[set_nr][0]
        profiles = self.sets[set_nr][1]
        pc_data = profiles[pfl_nr]
        if data_key=='alpha':
            return pc_data[:,0]
        elif data_key=='cl':
            return pc_data[:,1]
        elif data_key=='cd':
            return pc_data[:,2]
        elif data_key=='cm':
            return pc_data[:,3]
        elif data_key=='relative_thickness':
            return thickness[pfl_nr]
        else:
            raise KeyError('That variable does not exist')

    def __setitem__(self, key, val):

        '''Retrieve a value according to keys like '1.0.alpha' where the first is the data, the second is the profile and the third is the variable'''
        keys=key.split('.')
        set_nr = 1
        pfl_nr = 1
        data_key = 'alpha'
        if len(keys)==2:
            if len(self.sets)!=1:
                raise KeyError('Must specify the set in the key')
            set_nr = self.sets.keys()[0]
            pfl_nr = int(keys[0])
            data_key = keys[1]
        elif len(keys)==3:
            set_nr = int(keys[0])
            pfl_nr = int(keys[1])
            data_key = keys[2]
        else:
            raise KeyError('Key member nesting only 2 deep')
        thickness = self.sets[set_nr][0]
        profiles = self.sets[set_nr][1]
        pc_data = profiles[pfl_nr]

        if data_key=='relative_thickness':
            thickness[pfl_nr]=val
            return

        if len(val)!=len(pc_data):
            raise IndexError('The value is not the same size as the data')

        if data_key=='alpha':
            pc_data[:,0]=val
        elif data_key=='cl':
            pc_data[:,1]=val
        elif data_key=='cd':
            pc_data[:,2]=val
        elif data_key=='cm':
            pc_data[:,3]=val
        else:
            raise KeyError('That variable does not exist')

    def keys(self):
        '''Retrieve the keys of the different data'''
        retval=[]
        for I in self.sets.keys():
            thickness = self.sets[I][0]
            profiles = self.sets[I][1]
            for J in range(0,len(profiles)):
                retval.append(str(I)+'.'+str(J)+'.relative_thickness')
                retval.append(str(I)+'.'+str(J)+'.alpha')
                retval.append(str(I)+'.'+str(J)+'.cl')
                retval.append(str(I)+'.'+str(J)+'.cd')
                retval.append(str(I)+'.'+str(J)+'.cm')
        return retval

    def get_profile_size(self, set_id, profile_id):
        '''This will change the number of rows for a given profile'''
        if not set_id in self.sets.keys():
            raise KeyError('That set does not exist')
        thickness = self.sets[set_id][0]
        profiles = self.sets[set_id][1]
        if profile_id>=len(profiles):
            raise KeyError('That profile does not exist')
        return len(profiles[profile_id])

    def resize_profile(self, set_id, profile_id, row_nr):
        '''This will change the number of rows for a given profile'''
        if not set_id in self.sets.keys():
            raise KeyError('That set does not exist')
        thickness = self.sets[set_id][0]
        profiles = self.sets[set_id][1]
        if profile_id>=len(profiles):
            raise KeyError('That profile does not exist')
        old_data = profiles[profile_id]
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
        self.sets[set_id][1][profile_id]=new_data

    def get_profile_count(self, set_id):
        '''This will change the number of profiles in a given set'''
        if not set_id in self.sets.keys():
            raise KeyError('That set does not exist')
        return len(self.sets[set_id][1])

    def resize_set(self, set_id, polar_cnt, row_nr=0):
        '''This will change the number of profiles in a given set'''
        if not set_id in self.sets.keys():
            raise KeyError('That set does not exist')
        thickness = self.sets[set_id][0]
        profiles = self.sets[set_id][1]
        # do nothing if there is nothing to do
        if polar_cnt==len(profiles):
            return
        thickness.resize(polar_cnt, refcheck=False)
        if polar_cnt<len(profiles):
            profiles=profiles[:polar_cnt]
        else:
            for I in range(len(profiles), polar_cnt):
                profiles.append(np.zeros((row_nr,4)))
    
    def get_set_count(self):
        return len(self.sets)

    def add_set(self, polar_cnt=0, row_nr=0):
        I=1
        while I in self.sets.keys():
            I+=1
        polar=np.zeros((row_nr,4))
        polars=[polar]*polar_cnt
        thickness=np.zeros(polar_cnt)
        self.sets[I]=(thickness,polars)

    def get_data_str(self):

        retval=str(len(self.sets))+'\n'
        for set_id, set_data in self.sets.items():
            thickness = set_data[0]
            profiles  = set_data[1]
            retval+=str(len(profiles))+'\n'
            for I in range(0,len(profiles)):
                profile = profiles[I]
                thck = thickness[I]
                retval+=str(I+1)+' '+str(len(profile))+' '+str(thck)+'\n'
                for line in profile:
                    retval+='%25.17e %25.17e %25.17e %25.17e\n'%(line[0], line[1], line[2], line[3])
        return retval

    def __str__(self):
        return self.get_data_str()

    def read_file(self, filename):

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
    aefile = AEFile("tests/test_files/NREL_5MW_ae.txt")
    print (aefile.thickness(36))
    
    print (pcfile.CL(21,10)) # CL for thickness 21% and AOA=10deg
    #1.358
    print (pcfile.CD(21,10)) # CD for thickness 21% and AOA=10deg
    #0.0255
    print (pcfile.CM(21,10)) # CM for thickness 21% and AOA=10deg
    #-0.1103
