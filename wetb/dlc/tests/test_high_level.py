'''
Created on 09/10/2014

@author: MMPE
'''
import unittest
from wetb.dlc.high_level import DLCHighLevel
import os


class TestDLCHighLevel(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.dlc_hl = DLCHighLevel('test_files/DLC_test.xlsx')

    def test_variables(self):
        self.assertEqual(self.dlc_hl.vref, 50)
        self.assertEqual(os.path.realpath(self.dlc_hl.res_path), os.path.realpath(os.path.join(os.getcwd(), "test_files/res")))

    def test_sensor_info(self):
        self.assertEqual(list(self.dlc_hl.sensor_info().name), ['MxTB', 'MyTB', 'MxBR', 'PyBT', 'Power', 'Pitch', 'PitchBearing', 'Tip1TowerDistance', 'TipTowerDistance'])

    def test_sensor_info_filter(self):
        self.assertEqual(list(self.dlc_hl.sensor_info(['fatigue']).m), [4, 4, 10])


    def test_fatigue_distribution_pct(self):
        dlc, wsp, wdir = self.dlc_hl.fatigue_distribution()['12']
        self.assertEqual(dlc, 0.975)
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
        self.assertEqual(f, 'test_files\\res\\DLC12_IEC61400-1ed3\\dlc12_wsp04_wdir350_s3001.sel')
        self.assertEqual(h, .975 * .25 * 0.11002961306549919 / 2 * 20 * 365 * 24)

    def test_file_hour_lst_count(self):
        f, h = self.dlc_hl.file_hour_lst()[-1]
        self.assertEqual(f, 'test_files\\res\\DLC31_IEC61400-1ed3\\dlc31_wsp25_wdir000_s0000.sel')
        self.assertAlmostEqual(h, 0.0087201928 * 1 * (50 / 1100) * 20 * 365 * 24)


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


    def test_extremeload_sensors(self):
        self.dlc_hl = DLCHighLevel('test_files/DLC_test2.xlsx')



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
