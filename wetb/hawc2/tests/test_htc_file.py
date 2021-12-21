'''
Created on 17/07/2014

@author: MMPE
'''
import os
import unittest
from wetb.hawc2.htc_file import HTCFile, HTCLine


class TestHtcFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/htcfiles/')  # test file path

    def check_htc_file(self, f):

        with open(f) as fid:
            orglines = fid.read().strip().split("\n")

        htcfile = HTCFile(f, "../")
        newlines = str(htcfile).split("\n")
        htcfile.save(self.testfilepath + 'tmp.htc')
        # with open(self.testfilepath + 'tmp.htc') as fid:
        #    newlines = fid.readlines()

        for i, (org, new) in enumerate(zip(orglines, newlines), 1):
            def fmt(x): return x.strip().replace("\t", " ").replace(
                "  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            if fmt(org) != fmt(new):
                print("----------%d-------------" % i)
                print(fmt(org))
                print(fmt(new))
                self.assertEqual(fmt(org), fmt(new))
                break
                print()
        assert len(orglines) == len(newlines)

    def test_htc_files(self):
        for f in ['test3.htc']:
            self.check_htc_file(self.testfilepath + f)

    def test_htc_file_get(self):
        htcfile = HTCFile(self.testfilepath + "test3.htc", '../')
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
        self.assertEqual(htcfile['simulation']['logfile'][0],
                         './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile['simulation/logfile'][0],
                         './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
        self.assertEqual(htcfile['simulation.logfile'][0],
                         './logfiles/dlc12_iec61400-1ed3/dlc12_wsp10_wdir000_s1004.log')
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
        htcfile.simulation.name2 = ("value", 1)
        self.assertEqual(htcfile.simulation.name2[0], "value")
        self.assertEqual(htcfile.simulation.name2[1], 1)

    def test_htc_file_set_key2(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.simulation['name'] = "value"
        self.assertEqual(htcfile.simulation.name[0], "value")
        htcfile.simulation['name2'] = ("value", 1)
        self.assertEqual(htcfile.simulation.name2[0], "value")
        self.assertEqual(htcfile.simulation.name2[1], 1)

    def test_htc_file_del_key(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        del htcfile.simulation.logfile
        self.assertTrue("logfile" not in str(htcfile.simulation))
        try:
            del htcfile.hydro.water_properties.water_kinematics_dll
        except KeyError:
            pass

    def test_htc_file_delete(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertTrue("logfile" in str(htcfile.simulation))
        htcfile.simulation.logfile.delete()
        self.assertTrue("logfile" not in str(htcfile.simulation))

        self.assertTrue('newmark' in str(htcfile.simulation))
        htcfile.simulation.newmark.delete()
        with self.assertRaises(KeyError):
            htcfile.simulation.newmark

    def test_htcfile_setname(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        htcfile.set_name("mytest")
        self.assertEqual(os.path.relpath(htcfile.filename, self.testfilepath).replace("\\", "/"), r'../htc/mytest.htc')
        self.assertEqual(htcfile.simulation.logfile[0], './log/mytest.log')
        self.assertEqual(htcfile.output.filename[0], './res/mytest')

        htcfile.set_name("mytest", 'subfolder')
        self.assertEqual(os.path.relpath(htcfile.filename, self.testfilepath).replace(
            "\\", "/"), r'../htc/subfolder/mytest.htc')
        self.assertEqual(htcfile.simulation.logfile[0], './log/subfolder/mytest.log')
        self.assertEqual(htcfile.output.filename[0], './res/subfolder/mytest')

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
        htcfile.wind.mann.add_line("create_turb_parameters", [
                                   29.4, 1.0, 3.9, 1004, 1.0], "L, alfaeps, gamma, seed, highfrq compensation")
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[0], 29.4)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters[3], 1004)
        self.assertEqual(htcfile.wind.mann.create_turb_parameters.comments,
                         "L, alfaeps, gamma, seed, highfrq compensation")

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
    filename_u\t./turb/mann_l30.1_ae1.1000_g3.3_h0_16384x32x32_0.366x3.12x3.12_s0102u.turb;
    filename_v\t./turb/mann_l30.1_ae1.1000_g3.3_h0_16384x32x32_0.366x3.12x3.12_s0102v.turb;
    filename_w\t./turb/mann_l30.1_ae1.1000_g3.3_h0_16384x32x32_0.366x3.12x3.12_s0102w.turb;
    box_dim_u\t16384 0.3662;
    box_dim_v\t32 3.125;
    box_dim_w\t32 3.125;"""
        for a, b in zip(s.split("\n"), str(htcfile.wind.mann).split("\n")):
            self.assertEqual(a.strip(), b.strip())
        self.assertEqual(htcfile.wind.turb_format[0], 1)
        self.assertEqual(htcfile.wind.turb_format.comments, "0=none, 1=mann,2=flex")

    def test_add_turb_export(self):
        htc = HTCFile()
        htc.add_mann_turbulence(30.1, 1.1, 3.3, 102, False)
        htc.set_time(100, 700, 0.01)
        htc.add_turb_export()
        s = """begin turb_export;
  filename_u\texport_u.turb;
  filename_v\texport_v.turb;
  filename_w\texport_w.turb;
  samplefrq\t3;
  time_start\t100;
  nsteps\t60000.0;
  box_dim_v\t32 3.125;
  box_dim_w\t32 3.125;
end turb_export;"""
        for a, b in zip(s.split("\n"), str(htc.wind.turb_export).split("\n")):
            self.assertEqual(a.strip(), b.strip())

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
        htcfile = HTCFile(self.testfilepath + "test2.htc", '../')
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
        htcfile = HTCFile(self.testfilepath + "dlc14_wsp10_wdir000_s0000.htc", '../')
        self.assertEqual(htcfile.turbulence_files(), [
                         './turb/turb_wsp10_s0000u.bin', './turb/turb_wsp10_s0000v.bin', './turb/turb_wsp10_s0000w.bin'])

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
                  './control/risoe_controller_64.dll',
                  './control/generator_servo.dll',
                  './control/generator_servo_64.dll',
                  './control/mech_brake.dll',
                  './control/mech_brake_64.dll',
                  './control/servo_with_limits.dll',
                  './control/servo_with_limits_64.dll',
                  './control/towclearsens.dll',
                  './control/towclearsens_64.dll',
                  './data/user_shear.dat',
                  self.testfilepath.replace("\\", "/") + 'test.htc'
                  ]:
            try:
                input_files.remove(f)
            except ValueError:
                raise ValueError(f + " is not in list")
        self.assertFalse(input_files)

        htcfile = HTCFile(self.testfilepath + "DTU_10MW_RWT.htc")
        self.assertTrue('./control/wpdata.100' in htcfile.input_files())

    def test_input_files2(self):
        htcfile = HTCFile(self.testfilepath + "ansi.htc", '../')
        input_files = htcfile.input_files()
        self.assertTrue('./htc_hydro/ireg_airy_h6_t10.inp' in input_files)
        #

    def test_continue_in_files(self):
        htcfile = HTCFile(self.testfilepath + "continue_in_file.htc", ".")
        self.assertIn('main_body__31', htcfile.new_htc_structure.keys())
        self.assertIn(os.path.abspath(self.testfilepath + 'orientation.dat'),
                      [os.path.abspath(f) for f in htcfile.input_files()])
        self.assertIn('./data/NREL_5MW_st1.txt', htcfile.input_files())
        self.assertEqual(str(htcfile).count("exit"), 1)
        self.assertIn('filename\t./res/oc4_p2_load_case_eq;', str(htcfile).lower())

    def test_continue_in_files_autodetect_path(self):
        htcfile = HTCFile(self.testfilepath + "sub/continue_in_file.htc")
        self.assertIn('main_body__31', htcfile.new_htc_structure.keys())
        self.assertIn(os.path.abspath(self.testfilepath + 'orientation.dat'),
                      [os.path.abspath(f) for f in htcfile.input_files()])
        self.assertIn('./data/NREL_5MW_st1.txt', htcfile.input_files())
        self.assertEqual(str(htcfile).count("exit"), 1)
        self.assertIn('filename\t./res/oc4_p2_load_case_eq;', str(htcfile).lower())

    def test_tjul_example(self):
        htcfile = HTCFile(self.testfilepath + "./tjul.htc", ".")
        htcfile.save("./temp.htc")

    def test_ansi(self):
        htcfile = HTCFile(self.testfilepath + "./ansi.htc", '../')

    def test_file_with_BOM(self):
        htcfile = HTCFile(self.testfilepath + 'DLC15_wsp11_wdir000_s0000_phi000_Free_v2_visual.htc')
        self.assertEqual(str(htcfile)[0], ";")

    def test_htc_reset(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(htcfile.wind.wsp[0], 10)

    def test_htc_model_autodetect(self):
        htcfile = HTCFile(self.testfilepath + "test.htc")
        self.assertEqual(os.path.relpath(htcfile.modelpath, os.path.dirname(htcfile.filename)), "..")
        htcfile = HTCFile(self.testfilepath + "sub/test.htc")
        self.assertEqual(os.path.relpath(htcfile.modelpath, os.path.dirname(
            htcfile.filename)).replace("\\", "/"), "../..")
        self.assertRaisesRegex(ValueError, "Modelpath cannot be autodetected",
                               HTCFile, self.testfilepath + "missing_input_files.htc")

    def test_htc_model_autodetect_upper_case_files(self):
        htcfile = HTCFile(self.testfilepath + "../simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc")
        self.assertEqual(os.path.relpath(htcfile.modelpath, os.path.dirname(htcfile.filename)), "..")

    def test_open_eq_save(self):
        HTCFile(self.testfilepath + "test3.htc", "../").save(self.testfilepath + "tmp.htc")
        htcfile = HTCFile(self.testfilepath + "tmp.htc", "../")
        htcfile.save(self.testfilepath + "tmp.htc")
        self.assertEqual(str(htcfile).count("\t"), str(HTCFile(self.testfilepath + "tmp.htc", "../")).count("\t"))
        self.assertEqual(str(htcfile), str(HTCFile(self.testfilepath + "tmp.htc", "../")))

    def test_2xoutput(self):
        htc = HTCFile(self.testfilepath + "test_2xoutput.htc", "../")
        self.assertEqual(len(htc.res_file_lst()), 4)

    def test_access_section__1(self):
        htc = HTCFile(self.testfilepath + "test_2xoutput.htc", "../")
        assert htc.output__1.name_ == "output"

    def test_compare(self):
        htc = HTCFile(self.testfilepath + "test_cmp1.htc", "../")
        s = htc.compare(self.testfilepath + 'test_cmp2.htc')
        ref = """- begin subsection1;
- end subsection1;

+ begin subsection2;
+ end subsection2;

- begin section1;
- end section1;
- ;

+ alfa 2;
- alfa 1;

+ sensor1 1;
- sensor2 2;

+ begin section2;
+ end section2;
+ ;"""
        assert s.strip() == ref

    def test_pbs_file(self):
        htc = HTCFile(self.testfilepath + "../simulation_setup/DTU10MWRef6.0/htc/DTU_10MW_RWT.htc")
        assert os.path.relpath(htc.modelpath, self.testfilepath) == os.path.relpath(
            "../simulation_setup/DTU10MWRef6.0/")
        from wetb.hawc2.hawc2_pbs_file import JESS_WINE32_HAWC2MB
        htc.pbs_file(r"R:\HAWC2_tests\v12.6_mmpe3\hawc2\win32", JESS_WINE32_HAWC2MB)

    def test_pbs_file_inout(self):
        htc = HTCFile(self.testfilepath + "../simulation_setup/DTU10MWRef6.0_IOS/input/htc/DTU_10MW_RWT.htc")
        assert os.path.relpath(htc.modelpath, self.testfilepath) == os.path.relpath(
            "../simulation_setup/DTU10MWRef6.0_IOS/input")
        from wetb.hawc2.hawc2_pbs_file import JESS_WINE32_HAWC2MB
        # print(htc.pbs_file(r"R:\HAWC2_tests\v12.6_mmpe3\hawc2\win32",
        #                    JESS_WINE32_HAWC2MB, input_files=["./input/*"], output_files=['./output/*']))

    def test_htc_file_Path_object(self):
        from pathlib import Path
        htcfile = HTCFile(Path(self.testfilepath) / "test.htc")

    def test_htc_copy(self):
        htc = HTCFile(self.testfilepath + "test.htc")
        tower2 = htc.new_htc_structure.main_body.copy()
        tower2.name = "tower2"
        htc.new_htc_structure.add_section(tower2, allow_duplicate=True)
        assert htc.new_htc_structure.main_body__8 is tower2
        assert htc.new_htc_structure.main_body.name[0] == 'tower'
        assert htc.new_htc_structure.main_body__8.name[0] == 'tower2'
        ti2 = tower2.add_section(section_name='timoschenko_input',
                                 section=tower2.timoschenko_input.copy(), allow_duplicate=True)
        ti2.set = 3, 3
        assert tower2.timoschenko_input.set.values == [1, 2]
        assert tower2.timoschenko_input__2.set.values == [3, 3]

    def test_location(self):
        htc = HTCFile(self.testfilepath + "test.htc")
        assert htc.new_htc_structure.main_body__3.location() == 'test.htc/new_htc_structure/main_body__3'

    def test__call__(self):
        htc = HTCFile(self.testfilepath + "test.htc")
        assert htc.new_htc_structure.main_body(name='shaft').name.values[0] == 'shaft'
        assert htc.new_htc_structure.main_body.c2_def.sec(v0=3).values[0] == 3

    def test_jinja_tags(self):
        htc = HTCFile(self.testfilepath + "jinja.htc",
                      jinja_tags={'wsp': 12, 'logfilename': None, 'begin_step': 100})
        assert htc.wind.wsp[0] == 12
        assert 'logfile' not in htc.simulation
        assert htc.wind.wind_ramp_abs.values == [100, 101, 0, 1]
        assert htc.wind.wind_ramp_abs__2.values == [150, 151, 0, 1]
        htc = HTCFile(self.testfilepath + "jinja.htc",
                      jinja_tags={'wsp': 12, 'logfilename': 'test.log', 'begin_step': 100})
        assert htc.simulation.logfile[0] == 'test.log'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
