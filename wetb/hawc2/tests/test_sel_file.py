'''
Created on 17/07/2014

@author: MMPE
'''
import unittest
from wetb.hawc2.sel_file import SelFile, BINARY, ASCII
from datetime import datetime
import os
testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/hawc2io/')  # test file path

class TestSelFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)


    def test_sel_file_ascii(self):
        sf = SelFile(testfilepath + "Hawc2ascii.sel")
        self.assertEqual(sf.version_id, "HAWC2AERO 2.4w")
        self.assertEqual(sf.created, datetime(2013, 1, 24, 10, 2, 19))
        self.assertEqual(sf.result_file, "Hawc2ascii.dat")
        self.assertEqual(sf.scans, 800)
        self.assertEqual(sf.channels, 28)
        self.assertEqual(sf.no_sensors, 28)
        self.assertEqual(sf.duration, 20)
        self.assertEqual(sf.time, 20)
        self.assertEqual(sf.format, ASCII)
        self.assertEqual(sf.sensors[0], (1, 'Time', 's', 'Time'))
        self.assertEqual(sf.sensors[1], (2, 'WSP gl. coo.,Vy', 'm/s', 'Free wind speed Vy, gl. coo, of gl. pos    2.50,  -1.00, -47.50'))


    def test_sel_file_bin(self):
        sf = SelFile(testfilepath + "Hawc2bin.sel")
        self.assertEqual(sf.version_id, "HAWC2AERO 2.4w")
        self.assertEqual(sf.created, datetime(2013, 1, 24, 10, 4, 37))
        self.assertEqual(sf.result_file, "Hawc2bin.dat")
        self.assertEqual(sf.scans, 800)
        self.assertEqual(sf.channels, 28)
        self.assertEqual(sf.no_sensors, 28)
        self.assertEqual(sf.duration, 20)
        self.assertEqual(sf.time, 20)
        self.assertEqual(sf.format, BINARY)
        self.assertEqual(sf.sensors[0], (1, 'Time', 's', 'Time'))
        self.assertEqual(sf.sensors[1], (2, 'WSP gl. coo.,Vy', 'm/s', 'Free wind speed Vy, gl. coo, of gl. pos    2.50,  -1.00, -47.50'))
        self.assertEqual(sf.scale_factors[0], 6.25000E-04)
        self.assertEqual(sf.scale_factors[1], 5.65540E-02)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
