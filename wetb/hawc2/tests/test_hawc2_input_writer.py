import os

import pandas as pd
from pathlib import Path
import pytest

from wetb.hawc2.tests import test_files
from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2 import hawc2_input_writer
from tests.run_main import run_module_main


htc_base_dir = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/htc/'


@pytest.fixture
def h2writer():
    """HAWC2InputWriter with DTU 10 MW as base file, used in several tests"""
    htc_base_file = htc_base_dir + 'DTU_10MW_RWT.htc'
    return HAWC2InputWriter(htc_base_file)


def test_hawc2_writer_frompath(tmp_path):
    """Load from a path base, and write to path and string"""
    path = Path(htc_base_dir + 'DTU_10MW_RWT.htc')
    h2writer = HAWC2InputWriter(path)
    h2writer.write(tmp_path / 'test.htc')  # path
    h2writer.write((tmp_path / 'test.htc').as_posix())  # string


def test_hawc2_writer_noname_error(h2writer, tmp_path):
    """Should throw error if set_name is called without name"""
    wsp_lst = [4, 6, 8]
    df = pd.DataFrame({'wind.wsp': wsp_lst, 'Folder': ['tmp'] * 3})
    h2writer.from_pandas(df)
    with pytest.raises(KeyError):  # keyerror in write_all
        h2writer.write_all(tmp_path)


def test_pandas2htc_dottag(h2writer, tmp_path):
    """Load dlc from dict + pandas dataframe (wind speed tag with dot, name), write htc"""
    wsp_lst = [4, 6, 8]
    df = pd.DataFrame({'wind.wsp': wsp_lst, 'Name': ['c1', 'c2', 'c3'], 'Folder': ['tmp'] * 3})
    h2writer.from_pandas(df)
    h2writer.write_all(tmp_path)
    for i, wsp in enumerate(wsp_lst, 1):
        htc_path = tmp_path / ('tmp/c%d.htc' % i)
        htc = HTCFile(htc_path, modelpath=tmp_path.as_posix())
        assert htc.wind.wsp[0] == wsp
        assert htc.simulation.logfile[0] == ('./log/tmp/c%d.log' % i)
        assert htc.output.filename[0] == ('./res/tmp/c%d' % i)


def test_excel2htc(h2writer, tmp_path):
    """Load dlc from Excel (wind speed tag with dot, name), write to htc.
    Both string and pathlib.Path"""
    excel_path = os.path.dirname(test_files.__file__) + '/htc_input_table.xlsx'
    for i in range(2):
        h2writer.from_excel(excel_path if i else Path(excel_path))
        # print(h2writer.contents)
        h2writer.write_all(tmp_path)
        for i, wsp in enumerate([4, 6], 1):
            htc = HTCFile(tmp_path / ('tmp/a%d.htc' % i), modelpath=tmp_path.as_posix())
            assert htc.wind.wsp[0] == wsp
            assert htc.simulation.logfile[0] == ('./log/tmp/a%d.log' % i)
            assert htc.output.filename[0] == ('./res/tmp/a%d' % i)


def test_hawc2_writer_main():
    """Run the example in the hawc2_input_writer module"""
    run_module_main(hawc2_input_writer)


def test_hawc2_writer_custom(tmp_path):
    """Input writer with a custom 'set_'-style attribute"""
    htc_base_file = htc_base_dir + 'DTU_10MW_RWT.htc'

    class MyWriter(HAWC2InputWriter):
        def set_time(self, htc, **kwargs):
            htc.set_time(self.time_start, self.time_start + kwargs['time'])

    myWriter = MyWriter(htc_base_file, time_start=100)
    myWriter.write(tmp_path / 'w1.htc', **{'Name': 'w1', 'wind.wsp': 10, 'time': 600})
    htc = HTCFile(tmp_path / 'w1.htc', modelpath=tmp_path.as_posix())
    assert htc.simulation.time_stop[0] == 700
    assert htc.wind.wsp[0] == 10
    assert htc.output.filename[0] == './res/w1'


def test_CVF2pandas(h2writer):
    """Load dlc with constants, variables, functions format"""
    constants = {'simulation.time_stop': 100}
    variables = {'wind.wsp': [4, 6, 8],
                 'wind.tint': [0.1, 0.15, 0.2]}
    functions = {'Name': lambda x: 'sim_wsp' + str(x['wind.wsp']) + '_ti' + str(x['wind.tint'])}

    h2writer.from_CVF(constants, variables, functions)
    assert len(h2writer.contents) == 9
    assert set(list(h2writer.contents)) == set(['simulation.time_stop', 'wind.wsp', 'wind.tint', 'Name'])
