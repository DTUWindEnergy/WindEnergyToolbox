'''
Created on 01/08/2016

@author: MMPE
'''
import unittest
import os
from wetb.hawc2.bladeData import H2AeroBladeData, H2BladeData
tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestBladeData(unittest.TestCase):
    
    def testBladedata(self):
        #bd = H2AeroBladeData(tfp + "h2aero_tb/htc/h2aero.htc", '../')
        #bd = H2AeroBladeData(r'C:\mmpe\programming\python\Phd\pitot_tube\tests\test_files\swt36_107\h2a\htc\templates/swt3.6_107.htc', "../../")
        #print (bd.pcFile.chord(36))
        #bd = H2BladeData(r"C:\mmpe\HAWC2\models\SWT3.6-107\original_data\SWT4.0-130\dlc12_wsp06_wdir010_s18002.htc", ".")
        bd = H2BladeData(r"C:\mmpe\HAWC2\models\SWT3.6-107\htc/stat6/stat6_10.0_0.htc", "../../")
        #bd = H2BladeData(r"C:\mmpe\HAWC2\models\NREL 5MW reference wind turbine_play/htc/NREL_5MW_reference_wind_turbine_heli_step.htc", "../")
        if 1:
            bd.plot_geometry()
 
        


    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBladedata']
    unittest.main()
