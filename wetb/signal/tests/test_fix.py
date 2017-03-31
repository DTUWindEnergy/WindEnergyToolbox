'''
Created on 30. mar. 2017

@author: mmpe
'''
import os
import unittest
from wetb import gtsdf
from wetb.signal.fix._rotor_position import fix_rotor_position, find_fix_dt
from wetb.signal.filters._differentiation import differentiation


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')
import numpy as np
class TestFix(unittest.TestCase):


    def testRotorPositionFix(self):
        ds = gtsdf.Dataset(tfp+'azi.hdf5')
        sample_frq = 25
         
        #import matplotlib.pyplot as plt
        #print (find_fix_dt(ds.azi, sample_frq, ds.Rot_cor, plt))
        rp_fit = fix_rotor_position(ds.azi, sample_frq, ds.Rot_cor)
                  
        rpm_pos = differentiation(rp_fit)%180 / 360 * sample_frq * 60
        err_sum = np.sum((rpm_pos - ds.Rot_cor)**2)
         
        self.assertLess(err_sum,40)
        self.assertLess(np.sqrt(np.mean((((ds.azi-rp_fit)+180)%360-180)**2)),2)
        if 0:
            import matplotlib.pyplot as plt
            t = ds.Time-ds.Time[0]
            plt.plot(t, differentiation(ds.azi)%180 / 360 * sample_frq * 60, label='fit')
            plt.plot(t, ds.Rot_cor)
            plt.plot(t, differentiation(rp_fit)%180 / 360 * sample_frq * 60, label='fit')
            plt.ylim(10,16)
              
            plt.show()

 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRotorPositionFix']
    unittest.main()