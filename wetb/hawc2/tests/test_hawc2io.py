'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
import numpy as np
from wetb.hawc2.Hawc2io import ReadHawc2
import os
from wetb import gtsdf
from wetb.hawc2.Hawc2output import Hawc2Output
from wetb.hawc2.sensor_search import SensorSearch


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/hawc2io/')  # test file path


class TestHAWC2IO(unittest.TestCase):

    def test_doc_example(self):
        # if called with ReadOnly = 1 as
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        # no channels a stored in memory, otherwise read channels are stored for reuse

        # channels are called by a list
#        file([0,2,1,1])  => channels 1,3,2,2
        self.assertEqual(file([0, 2, 1, 1]).shape, (800, 4))

#        # if empty all channels are returned
#        file()  => all channels as 1,2,3,...
        self.assertEqual(file().shape, (800, 28))
#        file.t => time vector
        np.testing.assert_array_almost_equal(file.t, file([0])[:, 0])

    def test_read_binary_file(self):
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)

    def test_read_ascii_file(self):
        file = ReadHawc2(testfilepath + "Hawc2ascii", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)


def test_htc_line_in_gtsdf(self):
    res = Hawc2Output(
        os.path.join(testfilepath, "IEA15_htc_input_test.hdf5")
    )

    self.assertEqual(res().shape, (res.NrSc, res.NrCh))
    self.assertEqual(res(name="bea2").id, [3, 4, 5, 6, 7, 8])

    result = res(name="bea2", desc="angle speed", htc="pitch1")
    self.assertEqual(result.id, [4])
    self.assertEqual(result.shape, (res.NrSc, 1))

    self.assertEqual(
        res(htc=[" aero omega", " aero torque"]).id,
        [9, 10],
    )
    self.assertEqual(res(htc=" aero omega").name, ["Omega"])
    self.assertEqual(res(ChVec=[3, 5]).id, [3, 5])
    self.assertEqual(res(name="BEA 2").id, res(name="bea2").id)
    self.assertEqual(res(label="# tower base").id, [16, 17, 18])
    self.assertEqual(
        res(name="does-not-exist").shape,
        (res.NrSc, 0),
    )
    self.assertEqual(
        res(label="# tower base").id,
        res(label="towerbase").id,
    )

def test_sensor_search_validation(self):
    res = Hawc2Output(
        os.path.join(testfilepath, "IEA15_htc_input_test.hdf5")
    )

    self.assertEqual(
        res.get_sensor_id(name="bea2").tolist(),
        [3, 4, 5, 6, 7, 8],
    )
    self.assertEqual(res(0).id, [0])

    with self.assertRaisesRegex(
        ValueError,
        "ChVec must be one-dimensional",
    ):
        res([[0, 1]])

    with self.assertRaisesRegex(
        ValueError,
        "Channel number out of range",
    ):
        res([-1])

    with self.assertRaisesRegex(
        ValueError,
        "Channel number out of range",
    ):
        res([res.NrCh])

def test_sensor_search_metadata_validation(self):
    with self.assertRaisesRegex(
        ValueError,
        "No sensor metadata was provided",
    ):
        SensorSearch()

    with self.assertRaisesRegex(
        ValueError,
        "Inputs must have same length or be None",
    ):
        SensorSearch(
            names=["a", "b"],
            units=["m"],
        )

    sensors = SensorSearch(names=["a"])

    for kwargs, message in [
        ({"unit": "m"}, "Unit metadata is not available"),
        ({"desc": "load"}, "Description metadata is not available"),
        ({"htc": "output"}, "HTC_input metadata is not available"),
        ({"label": "tower"}, "Label metadata is not available"),
    ]:
        with self.subTest(kwargs=kwargs):
            with self.assertRaisesRegex(ValueError, message):
                sensors(**kwargs)

def test_sensor_search_get_sensor_id(self):
    sensors = SensorSearch(
        names=["tower", "blade", "tower top"],
    )

    result = sensors.get_sensor_id(name="tower")

    self.assertIsInstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([0, 2]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    test_htc_line_in_gtsdf()
    test_sensor_search_validation()
    test_sensor_search_metadata_validation()
    test_sensor_search_get_sensor_id()
