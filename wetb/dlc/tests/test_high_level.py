'''
Created on 09/10/2014

@author: MMPE
'''
import unittest
from wetb.dlc.high_level import DLCHighLevel, Weibull, Weibull_IEC
import os
import numpy as np

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestDLCHighLevel(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.dlc_hl = DLCHighLevel(testfilepath + 'DLC_test.xlsx')

    def test_variables(self):
        self.assertEqual(self.dlc_hl.vref, 50)
        self.assertEqual(os.path.realpath(self.dlc_hl.res_path), os.path.realpath(testfilepath + "res"))

    def test_sensor_info(self):
        self.assertEqual(list(self.dlc_hl.sensor_info().name), ['MxTB', 'MyTB', 'MxBR', 'PyBT', 'Power', 'Pitch', 'PitchBearing', 'Tip1TowerDistance', 'TipTowerDistance'])

    def test_sensor_info_filter(self):
        self.assertEqual(list(self.dlc_hl.sensor_info(['fatigue']).m), [4, 4, 10])


    def test_fatigue_distribution_pct(self):
        dlc, wsp, wdir = self.dlc_hl.fatigue_distribution()['12']
        self.assertEqual(dlc[12], 0.975)
        self.assertEqual(min(wsp.keys()), 4)
        self.assertEqual(max(wsp.keys()), 26)
        self.assertEqual(wsp[4], 0.11002961306549919)

    def test_fatigue_distribution_count(self):
        dlc, wsp, wdir = self.dlc_hl.fatigue_distribution()['31']
        #self.assertEqual(dlc, "#1000")
        self.assertEqual(min(wsp.keys()), 4)
        self.assertEqual(max(wsp.keys()), 25)
        self.assertEqual(wsp[4], "#1000")

    def test_file_hour_lst(self):
        f, h = self.dlc_hl.file_hour_lst()[0]
        self.assertEqual(os.path.abspath(f), os.path.abspath(testfilepath + 'res/dlc12_iec61400-1ed3/dlc12_wsp04_wdir350_s3001.sel'))
        self.assertEqual(h, .975 * .25 * 0.11002961306549919 / 2 * 20 * 365 * 24)

    def test_file_hour_lst_count(self):
        f, h = self.dlc_hl.file_hour_lst()[-1]
        self.assertEqual(os.path.abspath(f), os.path.abspath(testfilepath + 'res/dlc31_iec61400-1ed3/dlc31_wsp25_wdir000_s0000.sel'))
        self.assertAlmostEqual(h, 0.0087201928 * 1 * (50 / 1100) * 20 * 365 * 24)

    def test_file_dict_flex(self):
        dlc_hl = DLCHighLevel(testfilepath + 'DLC_test_flex.xlsx')
        file_lst = dlc_hl.files_dict()[12][4][350]["files"]
        self.assertEqual(len(file_lst),1)
        self.assertTrue(file_lst[0].endswith(".int"))


    def test_dlc_lst(self):
        self.assertEqual(self.dlc_hl.dlc_lst(), ['12', '13', '14', '31'])


    def test_dlc_lst_filter(self):
        self.assertEqual(self.dlc_hl.dlc_lst('F'), ['12', '31'])
        self.assertEqual(self.dlc_hl.dlc_lst('U'), ['12', '13', '31'])


    def test_psf(self):
        self.assertEqual(self.dlc_hl.psf()['31'], 1.3)

    def test_keys(self):
        for k in ['name', 'nr', 'description', 'unit', 'statistic', 'ultimate', 'fatigue', 'm', 'neql', 'bearingdamage', 'mindistance', 'maxdistance', 'extremeload']:
            self.assertTrue(k in self.dlc_hl.sensor_info().keys(), k)

    def test_fail_on_res_not_fount(self):
        # hack around FileNotFoundError not being in Python2.7
        try:
            self.dlc_hl = DLCHighLevel(testfilepath + 'DLC_test.xlsx',
                                       fail_on_resfile_not_found=True)
        except Exception as e:
            # FileNotFoundError on Py3.3+ inherits from IOError
            assert isinstance(e.__cause__, IOError)
#        self.assertRaises(FileNotFoundError, "Result files for dlc='12', wsp='6', wdir='-10' not found")




    def test_weibull_1(self):
        Vin = 0.0
        Vout = 100
        Vref = 50
        Vstep = .1
        shape_k = 2
        weibull = Weibull(Vref * 0.2, shape_k, Vin, Vout, Vstep)

        # total probability needs to be 1!
        p_tot = np.array([value for key, value in weibull.items()]).sum()
        self.assertAlmostEqual(p_tot, 1.0, 3)

    def test_weibull_2(self):
        Vin = 1.0
        Vout = 100
        Vref = 50
        Vstep = 2
        shape_k = 2
        weibull = Weibull(Vref * 0.2, shape_k, Vin, Vout, Vstep)
        # total probability needs to be 1!
        p_tot = np.array([value for key, value in weibull.items()]).sum()
        self.assertTrue(np.allclose(p_tot, 1.0))

    def test_weibull_IEC(self):
        Vref = 50
        np.testing.assert_array_almost_equal(Weibull_IEC(Vref, [4,6,8]), [ 0.11002961,  0.14116891,  0.15124155])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
