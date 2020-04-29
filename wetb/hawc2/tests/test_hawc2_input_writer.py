import pandas as pd
from wetb.hawc2.tests import test_files
from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
import os
import shutil
from wetb.hawc2.htc_file import HTCFile
import pytest
from wetb.hawc2 import hawc2_input_writer
from tests.run_main import run_module_main


path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/htc/'


def clean_htctmp():
    if os.path.isdir(path + "tmp"):
        shutil.rmtree(path + "tmp")


@pytest.yield_fixture(autouse=True)
def run_around_tests():
    clean_htctmp()
    yield
    clean_htctmp()


@pytest.fixture
def h2writer():
    htc_base_file = path + 'DTU_10MW_RWT.htc'
    return HAWC2InputWriter(htc_base_file)


def test_pandas2htc(h2writer):

    wsp_lst = [4, 6, 8]
    df = pd.DataFrame({'wind.wsp': wsp_lst, 'Name': ['c1', 'c2', 'c3'], 'Folder': ['tmp', 'tmp', 'tmp']})
    h2writer.from_pandas(df)
    h2writer.write_all(path)
    for i, wsp in enumerate(wsp_lst, 1):
        htc = HTCFile(path + "tmp/c%d.htc" % i)
        assert htc.wind.wsp[0] == wsp


def test_excel2htc(h2writer):
    h2writer.from_excel(os.path.dirname(test_files.__file__) + "/htc_input_table.xlsx")
    h2writer.write_all(path)
    for i, wsp in enumerate([4, 6], 1):
        htc = HTCFile(path + "tmp/a%d.htc" % i)
        assert htc.wind.wsp[0] == wsp


def test_hawc2_writer_main():
    run_module_main(hawc2_input_writer)


def test_hawc2_writer():
    htc_base_file = path + 'DTU_10MW_RWT.htc'

    # HAWC2InputWriter
    class MyWriter(HAWC2InputWriter):
        def set_time(self, htc, time, **_):
            htc.set_time(self.time_start, self.time_start + time)

    myWriter = MyWriter(htc_base_file, time_start=100)
    myWriter.write(path + "tmp/w1.htc", **{"Name": "w1", "wind.wsp": 10, "time": 600})
    htc = HTCFile(path + "tmp/w1.htc")
    assert htc.simulation.time_stop[0] == 700
    assert htc.wind.wsp[0] == 10


def test_CVF2pandas(h2writer):
    constants = {'simulation.time_stop': 100}
    variables = {'wind.wsp': [4, 6, 8],
                 'wind.tint': [0.1, 0.15, 0.2]}
    functions = {'Name': lambda x: 'sim_wsp' + str(x['wind.wsp']) + '_ti' + str(x['wind.tint'])}

    h2writer.from_CVF(constants, variables, functions)
    assert len(h2writer.contents) == 9
    assert set(list(h2writer.contents)) == set(['simulation.time_stop', 'wind.wsp', 'wind.tint', 'Name'])
