'''
Created on 30. mar. 2017

@author: mmpe
'''
import os
import unittest
from wetb import gtsdf
from wetb.signal.fix._rotor_position import fix_rotor_position,\
    find_polynomial_sample_length
from wetb.signal.filters._differentiation import differentiation


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')
import numpy as np
class TestFix(unittest.TestCase):


    def testRotorPositionFix(self):
        ds = gtsdf.Dataset(tfp+'azi.hdf5')
        sample_frq = 25
        print (find_polynomial_sample_length(ds.azi, sample_frq, ds.Rot_cor))
        rp_fit = fix_rotor_position(ds.azi, sample_frq, ds.Rot_cor, 500)
        
        rpm_pos = differentiation(rp_fit)%180 / 360 * sample_frq * 60
        
        self.assertLess(np.sum((rpm_pos - ds.Rot_cor)**2),50)
        if 0:
            import matplotlib.pyplot as plt
            plt.plot( differentiation(ds.azi)%180 / 360 * sample_frq * 60, label='fit')
            plt.plot(ds.Rot_cor)
            plt.plot( differentiation(rp_fit)%180 / 360 * sample_frq * 60, label='fit')
            plt.ylim(10,16)
            
            plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRotorPositionFix']
    unittest.main()