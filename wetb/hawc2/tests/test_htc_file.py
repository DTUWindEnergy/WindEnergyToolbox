'''
Created on 17/07/2014

@author: MMPE
'''
import os
import unittest

from datetime import datetime
from wetb.hawc2.htcfile import HTCFile

os.chdir(os.path.relpath(".", __file__))


import numpy as np



class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = "tests/test_files/"

    def test_htc_file(self):
        with open(self.testfilepath + 'test.htc') as fid:
            orglines = fid.readlines()

        htcfile = HTCFile(self.testfilepath + "test.htc")
        newlines = str(htcfile).split("\n")
        #htcfile.save(self.testfilepath + 'tmp.htc')
        #with open(self.testfilepath + 'tmp.htc') as fid:
        #    newlines = fid.readlines()

        for i, (org, new) in enumerate(zip(orglines, newlines), 1):
            fmt = lambda x : x.strip().replace("\t", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            if fmt(org) != fmt(new):
                print ("----------%d-------------" % i)
                print (fmt(org))
                print (fmt(new))
                self.assertEqual(fmt(org), fmt(new))
                break
                print ()
        assert len(orglines) == len(newlines)

    def test_htc_file_get(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(htcfile['simulation']['logfile'][0], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile['simulation/logfile'], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile.simulation.logfile, './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(float(htcfile.simulation.newmark.deltat), 0.02)
        self.assertEqual(htcfile.simulation.newmark, "deltat\t   0.02;\n")
        s = """time_stop\t    100;\nsolvertype\t   1"""
        self.assertEqual(str(htcfile.simulation)[:len(s)], s)


    def test_htc_file_set(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        time_stop = int(htcfile.simulation.time_stop)
        htcfile.simulation.time_stop = time_stop * 2
        self.assertEqual(int(htcfile.simulation.time_stop), 2 * time_stop)

    def test_htc_file_set_key(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.simulation.name = "value"
        self.assertEqual(htcfile.simulation.name, "value")

    def test_htc_file_del_key(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        del htcfile.simulation.logfile
        self.assertTrue("logfile" not in str(htcfile.simulation))
        try:
            del htcfile.hydro.water_properties.water_kinematics_dll
        except KeyError:
            pass

    def test_htcfile_setname(self):
        htcfile = HTCFile()
        htcfile.set_name("mytest")
        self.assertEqual(htcfile.filename, 'mytest.htc')



    def test_add_section(self):
        htcfile = HTCFile()
        htcfile.wind.add_section('mann')
        htcfile.wind.mann.create_turb_parameters = ("29.4 1.0 3.9 1004 1.0", "L, alfaeps, gamma, seed, highfrq compensation")
        self.assertTrue(htcfile.wind.mann.create_turb_parameters.startswith('29.4'))

    def test_add_mann(self):
        htcfile = HTCFile()
        htcfile.add_mann_turbulence(30.1, 1.1, 3.3, 102, False)
        s = """create_turb_parameters\t30.10 1.100 3.30 102 0;\tL, alfaeps, gamma, seed, highfrq compensation
filename_u\t./turb/turb_wsp10_s0102u.bin;
filename_v\t./turb/turb_wsp10_s0102v.bin;
filename_w\t./turb/turb_wsp10_s0102w.bin;
box_dim_u\t4096 1.4652;
box_dim_v\t32 3.2258;
box_dim_w\t32 3.2258;
std_scaling\t1.000000 0.800000 0.500000;"""
        for a, b in zip(s.split("\n"), str(htcfile.wind.mann).split("\n")):
            self.assertEqual(a, b)


    def test_sensors(self):
        htcfile = HTCFile()
        htcfile.set_name("test")
        htcfile.add_sensor('wind free_wind 1 0 0 -30')
        s = """filename\t./res/test;
general time;
wind free_wind 1 0 0 -30;"""
        for a, b in zip(s.split("\n"), str(htcfile.output).split("\n")):
            self.assertEqual(a, b)
        #print (htcfile)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
