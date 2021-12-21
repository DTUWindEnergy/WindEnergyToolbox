from tests import npt
import pytest
import numpy as np
from wetb.fast.fast_io import load_output, load_binary_output
import os

testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


def test_load_output():
    data, info = load_output(testfilepath + 'DTU10MW.out')
    npt.assert_equal(data[4, 3], 4.295E-04)
    npt.assert_equal(info['name'], "DTU10MW")
    npt.assert_equal(info['attribute_names'][1], "RotPwr")
    npt.assert_equal(info['attribute_units'][1], "kW")


def test_load_binary():
    data, info = load_output(testfilepath + 'test_binary.outb')
    npt.assert_equal(info['name'], 'test_binary')
    npt.assert_equal(info['description'], 'Modified by mwDeriveSensors on 27-Jul-2015 16:32:06')
    npt.assert_equal(info['attribute_names'][4], 'RotPwr')
    npt.assert_equal(info['attribute_units'][7], 'deg/s^2')
    npt.assert_almost_equal(data[10, 4], 138.822277739535)


def test_load_binary_buffered():
    # The old method was not using a buffer and was also memory expensive
    # Now use_buffer is set to true by default
    import numpy as np
    data, info = load_binary_output(testfilepath + 'test_binary.outb', use_buffer=True)
    data_old, info_old = load_binary_output(testfilepath + 'test_binary.outb', use_buffer=False)
    npt.assert_equal(info['name'], info_old['name'])
    np.testing.assert_array_equal(data[0, :], data_old[0, :])
    np.testing.assert_array_equal(data[-1, :], data_old[-1, :])


@pytest.mark.parametrize('fid,tol', [(2, 1), (3, 1e-4)])
@pytest.mark.parametrize('buffer', [True, False])
def test_load_bindary_fid(fid, tol, buffer):
    data2, info2 = load_output(testfilepath + '5MW_Land_BD_DLL_WTurb_fid04.outb')
    data, info = load_binary_output(testfilepath + '5MW_Land_BD_DLL_WTurb_fid%02d.outb' % fid, use_buffer=buffer)
    for k, v in info2.items():
        if k not in {'name', 'description'}:
            npt.assert_array_equal(info[k], v)
    r = data.max(0) - data.min(0) + 1e-20
    npt.assert_array_less(np.abs(data - data2).max(0), r * tol)


def test_load_output2():
    data, info = load_output(testfilepath + 'DTU10MW.out')
    npt.assert_equal(info['name'], "DTU10MW")
    npt.assert_equal(info['attribute_names'][1], "RotPwr")
    npt.assert_equal(info['attribute_units'][1], "kW")


def test_load_output3():
    # This file has an extra comment at the end
    data, info = load_output(testfilepath + 'FASTOut_Hydro.out')
    npt.assert_almost_equal(data[3, 1], -1.0E+01)
