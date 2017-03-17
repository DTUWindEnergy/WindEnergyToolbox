'''
Created on 17/07/2014

@author: MMPE
'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from io import open
from builtins import str
from builtins import zip
from future import standard_library
standard_library.install_aliases()
import os
import unittest

from datetime import datetime
from wetb.hawc2.htc_file import HTCFile, HTCLine



import numpy as np



class TestHtcFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/htcfiles/')  # test file path


    def check_htc_file(self, f):
 
        with open(f) as fid:
            orglines = fid.read().strip().split("\n")
 
        htcfile = HTCFile(f,"../")
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
        htcfile = HTCFile(self.testfilepath + "test3.htc",'../')
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
  
        htcfile.output.time = 10, 20
        self.assertEqual(htcfile.output.time[:2], [10, 20])
        self.assertEqual(str(htcfile.output.time), "time\t10 20;\n")
        htcfile.output.time = [11, 21]
        self.assertEqual(htcfile.output.time[:2], [11, 21])
        htcfile.output.time = "12 22"
        self.assertEqual(htcfile.output.time[:2], [12, 22])
  
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
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.set_name("mytest", htc_folder="htcfiles")
        self.assertEqual(os.path.relpath(htcfile.filename, self.testfilepath), r'mytest.htc')
        self.assertEqual(htcfile.simulation.logfile[0], './log/mytest.log')
        self.assertEqual(htcfile.output.filename[0], './res/mytest')
  
  
  
    def test_set_time(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.set_time(10, 20, 0.2)
        self.assertEqual(htcfile.simulation.time_stop[0], 20)
        self.assertEqual(htcfile.simulation.newmark.deltat[0], 0.2)
        self.assertEqual(htcfile.wind.scale_time_start[0], 10)
        self.assertEqual(htcfile.output.time[:2], [10, 20])
  
  
  
    def test_add_section(self):
        htcfile = HTCFile()
        htcfile.wind.add_section('mann')
        htcfile.wind.mann.add_line("create_turb_parameters", [29.4, 1.0, 3.9, 1004, 1.0], "L, alfaeps, gamma, seed, highfrq compensation")
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[0], 29.4)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[3], 1004)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters.comments, "L, alfaeps, gamma, seed, highfrq compensation")
        
    def test_add_section2(self):
        htcfile = HTCFile()
        htcfile.add_section('hydro')
        #self.assertEqual(str(htcfile).strip()[-5:], "exit;")
        
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.add_section('hydro')
        self.assertEqual(str(htcfile).strip()[-5:], "exit;")
        
          
  
    def test_add_mann(self):
        htcfile = HTCFile()
        htcfile.add_mann_turbulence(30.1, 1.1, 3.3, 102, False)
        s = """begin mann;
    create_turb_parameters\t30.1 1.1 3.3 102 0;\tL, alfaeps, gamma, seed, highfrq compensation
    filename_u\t./turb/mann_l30.1_ae1.10_g3.3_h0_4096x32x32_1.465x3.12x3.12_s0102u.turb;
    filename_v\t./turb/mann_l30.1_ae1.10_g3.3_h0_4096x32x32_1.465x3.12x3.12_s0102v.turb;
    filename_w\t./turb/mann_l30.1_ae1.10_g3.3_h0_4096x32x32_1.465x3.12x3.12_s0102w.turb;
    box_dim_u\t4096 1.4652;
    box_dim_v\t32 3.2258;
    box_dim_w\t32 3.2258;
    std_scaling\t1 0.8 0.5;"""
        for a, b in zip(s.split("\n"), str(htcfile.wind.mann).split("\n")):
            self.assertEqual(a.strip(), b.strip())
        self.assertEqual(htcfile.wind.turb_format[0], 1)
        self.assertEqual(htcfile.wind.turb_format.comments, "")
  
  
    def test_sensors(self):
        htcfile = HTCFile()
        htcfile.set_name("test")
        htcfile.output.add_sensor('wind', 'free_wind', [1, 0, 0, -30])
        s = """begin output;
    filename\t./res/test;
    general time;
    wind free_wind\t1 0 0 -30;"""
        for a, b in zip(s.split("\n"), str(htcfile.output).split("\n")):
            self.assertEqual(a.strip(), b.strip())
        #print (htcfile)
  
  
    def test_output_at_time(self):
        htcfile = HTCFile(self.testfilepath + "test2.htc",'../')
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
                  './res_eigen/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004/mode*.dat',
                  './launcher_test/ssystem_eigenanalysis.dat', './launcher_test/mode*.dat',
                  './res/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.sel',
                  './res/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.dat',
                  './res/rotor_check_inipos.dat',
                  './res/rotor_check_inipos2.dat']:
            try:
                output_files.remove(f)
            except ValueError:
                raise ValueError(f + " is not in list")
        self.assertFalse(output_files)
  
    def test_turbulence_files(self):
        htcfile = HTCFile(self.testfilepath + "dlc14_wsp10_wdir000_s0000.htc",'../')
        self.assertEqual(htcfile.turbulence_files(), ['./turb/turb_wsp10_s0000u.bin', './turb/turb_wsp10_s0000v.bin', './turb/turb_wsp10_s0000w.bin'])
  
    def test_input_files(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        input_files = htcfile.input_files()
        #print (htcfile.output)
        for f in ['./data/DTU_10MW_RWT_Tower_st.dat',
                  './data/DTU_10MW_RWT_Towertop_st.dat',
                  './data/DTU_10MW_RWT_Shaft_st.dat',
                  './data/DTU_10MW_RWT_Hub_st.dat',
                  './data/DTU_10MW_RWT_Blade_st.dat',
                  './data/DTU_10MW_RWT_ae.dat',
                  './data/DTU_10MW_RWT_pc.dat',
                  './control/risoe_controller.dll',
                  './control/generator_servo.dll',
                  './control/mech_brake.dll',
                  './control/servo_with_limits.dll',
                  './control/towclearsens.dll',
                  self.testfilepath.replace("\\","/") + 'test.htc'
                  ]:
            try:
                input_files.remove(f)
            except ValueError:
                raise ValueError(f + " is not in list")
        self.assertFalse(input_files)
  
    def test_input_files2(self):
        htcfile = HTCFile(self.testfilepath + "ansi.htc",'../')
        input_files = htcfile.input_files()
        self.assertTrue('./htc_hydro/ireg_airy_h6_t10.inp' in input_files)
        #
  
    def test_continue_in_files(self):
        htcfile = HTCFile(self.testfilepath + "continue_in_file.htc", ".")
        self.assertIn('main_body__31', htcfile.new_htc_structure.keys())
        self.assertIn(os.path.abspath(self.testfilepath + 'orientation.dat'), [os.path.abspath(f) for f in htcfile.input_files()])
        self.assertIn('./data/NREL_5MW_st1.txt', htcfile.input_files())
        self.assertEqual(str(htcfile).count("exit"), 1)
        self.assertIn('filename\t./res/oc4_p2_load_case_eq;', str(htcfile))
  
    def test_tjul_example(self):
        htcfile = HTCFile(self.testfilepath + "./tjul.htc", ".")
        htcfile.save("./temp.htc")
  
    def test_ansi(self):
        htcfile = HTCFile(self.testfilepath + "./ansi.htc",'../')
  
    def test_file_with_BOM(self):
        htcfile = HTCFile(self.testfilepath + 'DLC15_wsp11_wdir000_s0000_phi000_Free_v2_visual.htc')
        self.assertEqual(str(htcfile)[0], ";")
  
 
    def test_htc_reset(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(htcfile.wind.wsp[0], 10)
 
    def test_htc_model_autodetect(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(os.path.relpath(htcfile.modelpath,os.path.dirname(htcfile.filename)), "..")
        htcfile = HTCFile(self.testfilepath + "sub/test.htc")
        self.assertEqual(os.path.relpath(htcfile.modelpath,os.path.dirname(htcfile.filename)).replace("\\","/"), "../..")
        self.assertRaisesRegex(ValueError, "Modelpath cannot be autodetected", HTCFile, self.testfilepath + "test2.htc")
         
          


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
