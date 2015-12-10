'''
Created on 17/07/2014

@author: MMPE
'''
import os
import unittest

from datetime import datetime
from wetb.hawc2.htc_file import HTCFile, HTCLine

os.chdir(os.path.relpath(".", __file__))


import numpy as np



class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = "tests/test_files/htcfiles/"


    def check_htc_file(self, f):

        with open(f) as fid:
            orglines = fid.readlines()

        htcfile = HTCFile(f)
        newlines = str(htcfile).split("\n")
        htcfile.save(self.testfilepath + 'tmp.htc')
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

    def test_htc_files(self):
        for f in ['test3.htc']:
            self.check_htc_file(self.testfilepath + f)


    def test_htc_file_get(self):
        htcfile = HTCFile(self.testfilepath + "test3.htc")
        self.assertEqual(htcfile['simulation']['time_stop'][0], 200)
        self.assertEqual(htcfile['simulation/time_stop'][0], 200)
        self.assertEqual(htcfile['simulation.time_stop'][0], 200)
        self.assertEqual(htcfile.simulation.time_stop[0], 200)
        self.assertEqual(htcfile.dll.type2_dll.name[0], "risoe_controller")
        self.assertEqual(htcfile.dll.type2_dll__2.name[0], "risoe_controller2")
        s = """begin simulation;\n  time_stop\t200;"""
        self.assertEqual(str(htcfile.simulation)[:len(s)], s)

    def test_htc_file_get2(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(htcfile['simulation']['logfile'][0], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile['simulation/logfile'][0], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile['simulation.logfile'][0], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile.simulation.logfile[0], './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile.simulation.newmark.deltat[0], 0.02)

    def test_htc_file_set(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        time_stop = htcfile.simulation.time_stop[0]
        htcfile.simulation.time_stop = time_stop * 2
        self.assertEqual(htcfile.simulation.time_stop[0], 2 * time_stop)
        self.assertEqual(htcfile.simulation.time_stop.__class__, HTCLine)

    def test_htc_file_set_key(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.simulation.name = "value"
        self.assertEqual(htcfile.simulation.name[0], "value")

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
        htcfile.wind.mann.add_line("create_turb_parameters", [29.4, 1.0, 3.9, 1004, 1.0], "L, alfaeps, gamma, seed, highfrq compensation")
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[0], 29.4)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[3], 1004)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters.comments, "L, alfaeps, gamma, seed, highfrq compensation")

    def test_add_mann(self):
        htcfile = HTCFile()
        htcfile.add_mann_turbulence(30.1, 1.1, 3.3, 102, False)
        s = """begin mann;
  create_turb_parameters\t30.1 1.1 3.3 102 0;\tL, alfaeps, gamma, seed, highfrq compensation
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
        htcfile.output.add_sensor('wind', 'free_wind', [1, 0, 0, -30])
        s = """begin output;
  filename\t./res/test;
  general time;
  wind free_wind\t1 0 0 -30;"""
        for a, b in zip(s.split("\n"), str(htcfile.output).split("\n")):
            self.assertEqual(a, b)
        #print (htcfile)


    def test_output_at_time(self):
        htcfile = HTCFile(self.testfilepath + "test2.htc")
        self.assertTrue('begin output_at_time aero 15.0;' in str(htcfile))


    def test_output_files(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        output_files = htcfile.output_files()
        #print (htcfile.output)
        for f in ['./logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log',
                  './visualization/dlc12_wsp10_wdir000_s1004.hdf5',
                  './animation/structure_aero_control_turb.dat',
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_wsp10_wdir000_s1004_beam.dat',
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_wsp10_wdir000_s1004_body.dat',
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_wsp10_wdir000_s1004_struct.dat',
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_wsp10_wdir000_s1004_body_eigen.dat',
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_wsp10_wdir000_s1004_strc_eigen.dat',
                  './launcher_test/ssystem_eigenanalysis.dat', './launcher_test\\mode*.dat',
                  './res/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.sel',
                  './res/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.dat',
                  './res/rotor_check_inipos',
                  './res/rotor_check_inipos2']:
            try:
                output_files.remove(f)
            except ValueError:
                raise ValueError(f + " is not in list")
        self.assertFalse(output_files)

    def test_turbulence_files(self):
        htcfile = HTCFile(self.testfilepath + "dlc14_wsp10_wdir000_s0000.htc")
        self.assertEqual(htcfile.turbulence_files(), ['./turb/turb_wsp10_s0000u.bin', './turb/turb_wsp10_s0000v.bin', './turb/turb_wsp10_s0000w.bin'])

    def test_input_files(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        input_files = htcfile.input_files()
        #print (htcfile.output)
        for f in ['./data/DTU_10MW_RWT_Tower_st.dat',
                  './data/DTU_10MW_RWT_Towertop_st.dat',
                  './data/DTU_10MW_RWT_Shaft_st.dat',
                  './data/DTU_10MW_RWT_Hub_st.dat',
                  'data/2Bdown-rR1.08_blade_st.dat',
                  './data/2Bdown-rR1.08_ae.dat',
                  './data/2Bup_AEP095_pc.dat',
                  './control/risoe_controller.dll',
                  './control/generator_servo.dll',
                  './control/mech_brake.dll',
                  './control/servo_with_limits.dll',
                  './control/towclearsens.dll',
                  'tests/test_files/htcfiles/test.htc'
                  ]:
            try:
                input_files.remove(f)
            except ValueError:
                raise ValueError(f + " is not in list")
        self.assertFalse(input_files)

    def test_continue_in_files(self):
        htcfile = HTCFile(self.testfilepath + "continue_in_file.htc", ".")
        self.assertIn('main_body__31', htcfile.new_htc_structure.keys())
        self.assertIn('tests/test_files/htcfiles\\.\\./orientation.dat', htcfile.input_files())
        self.assertIn('tests/test_files/htcfiles\\.\\./orientation.dat', htcfile.input_files())
        self.assertIn('./data/NREL_5MW_st1.txt', htcfile.input_files())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
