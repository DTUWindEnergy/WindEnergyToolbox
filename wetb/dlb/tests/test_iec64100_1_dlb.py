from wetb.dlb import iec64100_1


import pytest
from wetb.dlb.hawc2_iec_dlc_writer import HAWC2_IEC_DLC_Writer

import os
from wetb.hawc2.tests import test_files
import shutil
from tests import npt, run_main
from wetb.hawc2.htc_file import HTCFile
from tests.run_main import run_module_main
from wetb.dlb.iec64100_1 import DTU_IEC64100_1_Ref_DLB

path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/htc/tmp/'


def clean_up():
    if os.path.isdir(path):
        shutil.rmtree(path)


@pytest.yield_fixture(autouse=True)
def run_around_tests():
    clean_up()
    yield
    clean_up()


@pytest.fixture
def writer():
    return HAWC2_IEC_DLC_Writer(path + '../DTU_10MW_RWT.htc', diameter=127)


def test_main():
    run_main.run_module_main(iec64100_1)


def test_DLC12(writer):
    dlc12 = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)['DLC12']
    assert len(dlc12) == 216  # 12 wsp, 3 wdir, 6 seeds
    writer.from_pandas(dlc12[::24][:2])
    writer.write_all(path)
    npt.assert_array_equal(sorted(os.listdir(path + "DLC12")),
                           ['DLC12_wsp04_wdir350_s1001.htc', 'DLC12_wsp06_wdir000_s1101.htc'])
    htc = HTCFile(path + "DLC12/DLC12_wsp04_wdir350_s1001.htc")
    assert htc.wind.wsp[0] == 4
    npt.assert_array_equal(htc.wind.windfield_rotations.values, [-10, 0, 0])
    assert htc.wind.turb_format[0] == 1
    assert htc.wind.mann.create_turb_parameters[3] == 1001


def test_DLC21(writer):
    dlc = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)['DLC21']
    assert len(dlc) == 144  # 12 wsp, 3 wdir, 4 seeds
    writer.from_pandas(dlc[::16][:2])
    writer.write_all(path)
    npt.assert_array_equal(sorted(os.listdir(path + "DLC21")),
                           ['DLC21_wsp04_wdir350_s1001.htc', 'DLC21_wsp06_wdir000_s1101.htc'])
    htc = HTCFile(path + "DLC21/DLC21_wsp04_wdir350_s1001.htc")
    assert htc.wind.wsp[0] == 4
    npt.assert_array_equal(htc.wind.windfield_rotations.values, [-10, 0, 0])
    assert htc.wind.turb_format[0] == 1
    assert htc.wind.mann.create_turb_parameters[3] == 1001
    assert htc.dll.get_subsection_by_name('generator_servo', 'name').init.constant__7.values == [7, 110]


def test_DLC22y(writer):
    dlc = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)['DLC22y']
    assert len(dlc) == 276  # 12 wsp, 23 wdir, 1 seeds
    writer.from_pandas(dlc[::24][:2])
    writer.write_all(path)
    npt.assert_array_equal(sorted(os.listdir(path + "DLC22y")),
                           ['DLC22y_wsp04_wdir015_s1001.htc', 'DLC22y_wsp06_wdir030_s1101.htc'])
    htc = HTCFile(path + "DLC22y/DLC22y_wsp04_wdir015_s1001.htc")
    assert htc.wind.wsp[0] == 4
    npt.assert_array_equal(htc.wind.windfield_rotations.values, [15, 0, 0])
    assert htc.wind.turb_format[0] == 1
    assert htc.wind.mann.create_turb_parameters[3] == 1001
