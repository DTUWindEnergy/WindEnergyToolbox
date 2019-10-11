'''
Created on 11/10/2019

@author: JYLI
'''
from wetb.hawc2 import sel_file
from pathlib import Path
import numpy as np
import os


def ResultFile(filename):
    '''
    A factory function that returns an instance of a ResultIO class.
    The type of ResultIO class depends on the result file format.
    Available formats:
    - HAWC2_BINARY
    - HAWC2_ASCII (not implemented)
    - GTSDF (not implemented)
    - FLEX (not implemented)
    '''
    for cls in ResultIO.__subclasses__():
        if cls.matching_format(filename):
            return cls(filename)
    raise ValueError
    
    
    
    
class ResultIO(object):
    '''
    base class for a HAWC2 result reader/writer.
    '''
    def __init__(self, filename):
        self.filename = Path(filename)
        if self.filename.suffix.lower() in ['.sel', '.int', '.hdf5']:
            self.filename = self.filename.with_suffix('')
        
        self.meta = self.read_meta(filename)
        self._data = None
        
    ## Reading methods
    def read_meta(self, filename):
        pass
        
                
    @property
    def data(self):
        '''
        Lazily loads the result data.
        '''
        if self._data is None:
            self._data = self.read()
        return self._data
        

    def read(self, channels=None):
        pass
        
        
    ## Writing methods
    @data.setter
    def data(self, value):
        self._data = value
        
    def write(self, filename):
        pass
        
        
    def add_channel(self, new_data, name, unit, desc):
        assert len(new_data) == self.meta['NrSc'] 
        new_data = new_data.reshape(-1, 1)
        
        self.data = np.concatenate([self.data, new_data], axis=1)
        self._add_channel_meta(new_data, name, unit, desc)
        
    
    def _add_channel_meta(self, new_data, name, unit, desc):
        pass
        
            
    def remove_channel(self, channels):
        pass
    
    
    
class BinaryFile(ResultIO):
    
    def matching_format(filename):
        if os.path.isfile(filename + ".sel"):
            if sel_file.SelFile(filename + '.sel').format == 'BINARY':
                return True
        return False
        
        
    def read_meta(self, filename):
        meta = {}
        sf              = sel_file.SelFile(filename + '.sel')
        
        meta['format']        = sf.format
        meta['version']       = sf.version_id
        meta['created']       = sf.created
        meta['NrSc']          = sf.scans
        meta['NrCh']          = sf.no_sensors
        meta['duration']      = sf.duration
        meta['ChInfo']        = [(b, c, d) for (a, b, c, d) in sf.sensors]
        meta['scale_factors'] = sf.scale_factors
        
        return meta

      
    def read(self, channels=None):
        if not channels:
            channels = range(0, self.meta['NrCh'] )
            
        with open(self.filename.as_posix() + '.dat', 'rb') as fid:
            data = np.zeros((self.meta['NrSc'] , len(channels)))
            j = 0
            for i in channels:
                fid.seek(i * self.meta['NrSc']  * 2, 0)
                data[:, j] = np.fromfile(fid, 'int16', self.meta['NrSc'] ) * self.meta['scale_factors'][i]
                j += 1
        return data
    
        
    def write(self, filename):
        filename = Path(filename)
        sel_file.save(filename.as_posix() + '.sel', self.meta['version'], self.meta['created'], self.meta['NrSc'], self.meta['NrCh'], self.meta['duration'], self.meta['ChInfo'], self.meta['scale_factors'])
        
        ChVec = range(0, self.meta['NrCh'])
        with open(filename.as_posix() + '.dat', 'wb') as fid:
            for i in ChVec:
                scale = abs(self.data[:, i]).max()/32000
                if scale == 0: scale = 1
                
                this_data = (self.data[:, i]/scale).round().astype('int16')
                
                fid.write(this_data.tobytes())
        
        
    def _add_channel_meta(self, new_data, name, unit, desc):

        self.meta['ChInfo'].append((name, unit, desc))
        self.meta['NrCh'] += 1
        self.meta['scale_factors'] = np.append(self.meta['scale_factors'], abs(new_data).max()/32000)  




class ASCIIFile(ResultIO):
    
    def matching_format(filename):
        if os.path.isfile(filename + ".sel"):
            if sel_file.SelFile(filename + '.sel').format == 'ASCII':
                return True
        return False
        
        
    def read_meta(self, filename):
        meta = {}
        sf              = sel_file.SelFile(filename + '.sel')
        
        meta['format']        = sf.format
        meta['version']       = sf.version_id
        meta['created']       = sf.created
        meta['NrSc']          = sf.scans
        meta['NrCh']          = sf.no_sensors
        meta['duration']      = sf.duration
        meta['ChInfo']        = [(b, c, d) for (a, b, c, d) in sf.sensors]
        
        return meta

      
    def read(self, channels=None):
        if not channels:
            channels = range(0, self.meta['NrCh'])
        temp = np.loadtxt(self.filename.as_posix() + '.dat', usecols=channels)
        return temp.reshape((self.meta['NrSc'], len(channels)))

        return data
    
        
    def write(self, filename):
        sel_file.save(filename + '.sel', self.meta['version'], self.meta['created'], self.meta['NrSc'], self.meta['NrCh'], self.meta['duration'], self.meta['ChInfo'])
        np.savetxt(filename + '.dat', self.data)
        
        
    def _add_channel_meta(self, new_data, name, unit, desc):

        self.meta['ChInfo'].append((name, unit, desc))
        self.meta['NrCh'] += 1

    
