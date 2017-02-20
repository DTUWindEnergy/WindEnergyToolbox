'''
Created on 20. feb. 2017

@author: mmpe
'''
import unittest
from wetb import gtsdf
from wetb.wind.dir_mapping import wsp_dir2uv
from wetb.wind.turbulence.mann_parameters import fit_mann_model_spectra
import numpy as np
from wetb.wind.turbulence.turbulence_spectra import spectra
import os
tfp = os.path.join(os.path.dirname(__file__), "test_files/") + "/"

class TestMannParameters(unittest.TestCase):

   
    def test_estimate_mann_parameters1(self):
        """Example of fitting Mann parameters to a time series"""
        
        ds = gtsdf.Dataset(tfp+"WspDataset.hdf5")#'unit_test/test_files/wspdataset.hdf5')
        f = 35
        u, v = wsp_dir2uv(ds.Vhub_85m, ds.Dir_hub_)
    
        u_ref = np.mean(u)
        u -= u_ref
    
        sf = f / u_ref
        plt=False
        ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v), plt = plt)
        
        self.assertAlmostEqual(ae, 0.03, 3)
        self.assertAlmostEqual(L, 16.20, 2)
        self.assertAlmostEqual(G, 2.47, 2)
    

    def test_estimate_mann_parameters2(self):
        """Example of fitting Mann parameters to a "series" of a turbulence box"""
        l = 16384
        nx = 8192
        ny, nz = 8, 8
        sf = (nx / l)
        fn = tfp + "turb/h2a8192_8_8_16384_32_32_0.15_10_3.3%s.dat"
        u, v, w = [np.fromfile(fn % uvw, np.dtype('<f'), -1).reshape(nx , ny * nz) for uvw in ['u', 'v', 'w']]
        plt=False
        ae, L, G = fit_mann_model_spectra(*spectra(sf, u, v, w), plt=plt)
        self.assertAlmostEqual(ae, 0.15, delta=0.01)
        self.assertAlmostEqual(L, 10, delta=0.3)
        self.assertAlmostEqual(G, 3.3, delta=0.06)
    
    
            
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_estimate_mann_parameters']
    unittest.main()