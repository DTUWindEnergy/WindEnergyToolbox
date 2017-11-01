'''
Created on 3. maj 2017

@author: mmpe
'''
import os
import unittest

from wetb.hawc2.blade import H2Blade
import numpy as np

tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class Test(unittest.TestCase):


#     def testBladeInfo(self):
#         bi = H2Blade(tfp + "simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc")
#         if 0:
#             import matplotlib.pyplot as plt
#             print (dir(bi))
#             #print (bi.radius_s())
#             plt.plot(bi.radius_s(), bi.twist())
#             plt.plot(bi.c2def[:,2], bi.c2def[:,3])
#             x = np.linspace(0,1,1000)
#             plt.plot(bi.blade_radius*x, bi.c2nd(x)[:,3])
#             plt.show()

            
    def testBladeInfo_AE(self):
        bi = H2Blade(None, tfp + "NREL_5MW_ae.txt", tfp + "NREL_5MW_pc.txt")
        self.assertEqual(bi.thickness(32), 23.78048780487805)
        self.assertEqual(bi.chord(32), 3.673)
        self.assertEqual(bi.pc_set_nr(32), 1)

    def testBladeInfo_PC_AE(self):
        bi = H2Blade(None, tfp + "NREL_5MW_ae.txt", tfp + "NREL_5MW_pc.txt")
        self.assertEqual(bi.CL(36, 10), 1.358)
        self.assertEqual(bi.CD(36, 10), 0.0255)
        self.assertEqual(bi.CM(36, 10), -0.1103)
            
#     def test_curved_length2radius(self):
#         bi = H2BladeInfo(tfp + "simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc")
#         np.testing.assert_array_equal(bi.xyztwist(86.4979, curved_length=True), bi.xyztwist(86.3655))
        
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBladeInfo']
    unittest.main()