import numpy as np
import os
import pandas as pd
import shutil
import unittest
from wetb.dlb.hawc2_iec_dlc_writer import HAWC2_IEC_DLC_Writer 
from wetb.dlb.iec61400_1 import DTU_IEC61400_1_Ref_DLB
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.tests import test_files

class Test_DTU_IEC61400_1_Ref_DLB(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.dlb = DTU_IEC61400_1_Ref_DLB(iec_wt_class='1A',
                                         D=178.3,
                                         z_hub=119,
                                         Vin=4,
                                         Vr=11,
                                         Vout=25)
        
        h2writer = HAWC2_IEC_DLC_Writer(os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc').from_pandas(cls.dlb)
        
        if os.path.isdir('./tmp'):
            shutil.rmtree('./tmp')
        # Write only 1 file per each DLC to keep it light
        h2writer.contents = pd.concat([h2writer.contents[h2writer.contents['DLC'] == dlc].iloc[0:1] for dlc in set(h2writer.contents['DLC'])])
        h2writer.write_all('./tmp')
    
    def setUp(self):
        unittest.TestCase.setUp(self)
    
    @classmethod
    def tearDownClass(cls):
        super(Test_DTU_IEC61400_1_Ref_DLB, cls).tearDownClass()
        
    def test_number_of_sims(self):
        # Take example from report "Design Load Basis for onshore turbines - Revision 00"
        sims = self.dlb.to_pandas()
        n_sims_dict = {'DLC12': 216,
                       'DLC13': 216,
                       'DLC14': 3,
                       'DLC15': 48,
                       'DLC21': 144,
                       'DLC22b': 144,
                       'DLC22p': 96,
                       'DLC22y': 276,
                       'DLC23': 9,
                       'DLC24': 72,
                       'DLC31': 3,
                       'DLC32': 16,
                       'DLC33': 16,
                       'DLC41': 3,
                       'DLC42': 18,
                       'DLC51': 36,
                       'DLC61': 12,
                       'DLC62': 24,
                       'DLC63': 12,
                       'DLC64': 204, # 1 wsp more due to rounding of 0.7 * Vref
                       'DLC71': 96,
                       'DLC81': 12}
        for dlc, n_sims in n_sims_dict.items():
            self.assertEqual(len(sims[sims['DLC'] == dlc]), n_sims, dlc)
                
    ### Turbulence models ###
    
    def test_NoTurb(self):
        htc = HTCFile('./tmp/DLC14/DLC14_wsp09_wdir000.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.wind.turb_format[-1], 0)
    
    def test_ConstantTurb(self):
        htc = HTCFile('./tmp/DLC61/DLC61_wsp50_wdir352_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        ti = 0.11
        self.assertEqual(htc.wind.tint[-1], ti)
    
    def test_NTM(self):
        htc = HTCFile('./tmp/DLC12/DLC12_wsp04_wdir350_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        ti = 0.16 * (0.75 + 5.6 / 4)
        self.assertEqual(htc.wind.tint[-1], ti)
        
    def test_ETM(self):
        htc = HTCFile('./tmp/DLC13/DLC13_wsp04_wdir350_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        ti = 2 * 0.16 * (0.072 * (0.2 * 50 / 2 + 3) * (4 / 2 - 4) + 10) / 4
        self.assertEqual(htc.wind.tint[-1], ti)
    
    ### Shear profiles ###
    
    def test_NWP(self):
        htc = HTCFile('./tmp/DLC12/DLC12_wsp04_wdir350_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.wind.shear_format.values, [3, 0.2])
        
    def test_EWS(self):
        htc = HTCFile('./tmp/DLC15/DLC15_wsp04_wdir000_ews++.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.wind.shear_format.values, [3, 0.2])
        A = (2.5 + 0.2 * 6.4 * 0.16 * (0.75 * 4 + 5.6) * (178.3 / 42) ** 0.25) / 178.3
        self.assertEqual(htc.wind.iec_gust.values, ['ews', A, 0, 100, 12])
    
    ### Gusts ###
            
    def test_ECD(self):
        htc = HTCFile('./tmp/DLC14/DLC14_wsp09_wdir000.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        V_cg = 15  
        theta_cg = 80  
        T = 10  
        self.assertEqual(htc.wind.iec_gust.values, ['ecd', V_cg, theta_cg, 100, T])        
    
    def test_EOG(self):
        htc = HTCFile('./tmp/DLC23/DLC23_wsp09_wdir000_t0.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        V_gust = 3.3 * 0.16 * (0.75 * 9 + 5.6) / (1 + 0.1 * 178.3 / 42) 
        T_gust = 10.5  
        self.assertEqual(htc.wind.iec_gust.values, ['eog', V_gust, 0, 100, T_gust])
        
    def test_EDC(self):
        htc = HTCFile('./tmp/DLC33/DLC33_wsp04_wdir000_-_t0.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        phi = 4 * (np.rad2deg(np.arctan(0.16 * (0.75 * 4 + 5.6) / 4 / (1 + 0.1 * 178.3 / 42)))) 
        T = 10  
        self.assertEqual(htc.wind.iec_gust.values, ['edc', 0, -phi, 100, T])
    
    ### Faults ###    
    
    def test_StuckBlade(self):
        htc = HTCFile('./tmp/DLC22b/DLC22b_wsp04_wdir000_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll__4.init.constant__9[-1], 0.1)
        self.assertEqual(htc.dll.type2_dll__4.init.constant__10[-1], 0)
    
    def test_PitchRunaway(self):
        htc = HTCFile('./tmp/DLC22p/DLC22p_wsp11_wdir000_s1301.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll__4.init.constant__8[-1], 100 + 10)
    
    def test_GridLoss(self):
        htc = HTCFile('./tmp/DLC21/DLC21_wsp04_wdir350_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll__2.init.constant__7[-1], 100 + 10)
    
    ### Operation ###
    
    def test_StartUp(self):
        htc = HTCFile('./tmp/DLC31/DLC31_wsp04_wdir000.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll.init.constant__24[-1], 100)
    
    def test_ShutDown(self):
        htc = HTCFile('./tmp/DLC41/DLC41_wsp04_wdir000.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll.init.constant__26[-1], 100)
        self.assertEqual(htc.dll.type2_dll.init.constant__28[-1], 1)
    
    def test_EmergencyShutDown(self):
        htc = HTCFile('./tmp/DLC51/DLC51_wsp09_wdir000_s1201.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll.init.constant__26[-1], 100)
        self.assertEqual(htc.dll.type2_dll.init.constant__28[-1], 2)
    
    def test_Parked(self):
        htc = HTCFile('./tmp/DLC64/DLC64_wsp04_wdir352_s1001.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.dll.type2_dll.init.constant__24[-1], 700 + 1)
    
    def test_RotorLocked(self):
        htc = HTCFile('./tmp/DLC81/DLC81_wsp18_wdir352_s1701.htc', modelpath='../../hawc2/tests/test_files/simulation_setup/DTU10MWRef6.0')
        self.assertEqual(htc.new_htc_structure.orientation.relative__2[htc.new_htc_structure.orientation.relative__2.keys()[-2]].values, [0, 0, 180])
        self.assertEqual(htc.new_htc_structure.constraint.bearing3.omegas[0], 0)
            
if __name__ == "__main__":
    unittest.main()

